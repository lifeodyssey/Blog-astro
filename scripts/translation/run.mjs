#!/usr/bin/env node

import { execFileSync } from 'node:child_process'
import { existsSync, readFileSync, writeFileSync } from 'node:fs'
import { extname, join, dirname } from 'node:path'
import { fileURLToPath } from 'node:url'
import process from 'node:process'

const __filename = fileURLToPath(import.meta.url)
const __dirname = dirname(__filename)

const SUPPORTED_LANGS = new Set(['zh', 'en', 'ja'])
const CLASS_TYPES = new Set(['tech_share', 'personal_note', 'diary_life'])
const SOURCE_EXT_RE = /\.(md|mdx)$/i
const GENERATED_LANG_SUFFIX_RE = /\.(zh|en|ja)\.(md|mdx)$/i

const DEFAULT_TARGET_MATRIX = {
  zh: ['en', 'ja'],
  ja: ['zh', 'en'],
  en: ['zh', 'ja'],
}

let styleCorpusCache = null

async function main() {
  const args = parseArgs(process.argv.slice(2))
  const dryRun = toBoolean(args['dry-run']) || false

  const config = {
    dryRun,
    defaultSourceLang: normalizeLang(getEnv('DEFAULT_SOURCE_LANG') || 'zh') || 'zh',
    targetMatrix: parseTargetMatrix(getEnv('TRANSLATION_TARGET_MATRIX_JSON')),
    classificationThreshold: clampNumber(Number(getEnv('CLASSIFICATION_CONFIDENCE_THRESHOLD') || '0.75'), 0, 1),
    verificationMinScore: clampNumber(Number(getEnv('VERIFICATION_MIN_SCORE') || '80'), 0, 100),
    variantsPerProvider: clampInt(Number(getEnv('VARIANTS_PER_PROVIDER') || '2'), 1, 4),
    reviewMaxRevisions: clampInt(Number(getEnv('REVIEW_MAX_REVISIONS') || '1'), 0, 3),
    updateExistingTranslations: !isExplicitFalse(getEnv('UPDATE_EXISTING_TRANSLATIONS')),
    overwriteManualTranslations: toBoolean(getEnv('OVERWRITE_MANUAL_TRANSLATIONS')) || false,
    failOnError: toBoolean(getEnv('TRANSLATION_FAIL_ON_ERROR')) || false,
  }

  const modelConfig = {
    anthropic: {
      apiBaseUrl: getEnv('ANTHROPIC_API_BASE_URL') || 'https://api.anthropic.com/v1/messages',
      apiKey: getEnv('ANTHROPIC_API_KEY'),
      classifyModel: getEnv('ANTHROPIC_MODEL_CLASSIFY') || getEnv('ANTHROPIC_MODEL') || 'claude-3-5-sonnet-latest',
      translateModel: getEnv('ANTHROPIC_MODEL_TRANSLATE') || getEnv('ANTHROPIC_MODEL') || 'claude-3-5-sonnet-latest',
      reviseModel: getEnv('ANTHROPIC_MODEL_REVISE') || getEnv('ANTHROPIC_MODEL_TRANSLATE') || getEnv('ANTHROPIC_MODEL') || 'claude-3-5-sonnet-latest',
    },
    gemini: {
      apiBaseUrl: getEnv('GEMINI_API_BASE_URL'),
      apiKey: getEnv('GEMINI_API_KEY'),
      translateModel: getEnv('GEMINI_MODEL_TRANSLATE') || getEnv('GEMINI_MODEL') || 'gemini-2.0-flash',
      reviewModel: getEnv('GEMINI_MODEL_REVIEW') || getEnv('GEMINI_MODEL') || 'gemini-2.0-flash',
    },
  }

  const anthropicClient = createAnthropicClient({
    dryRun: config.dryRun,
    apiBaseUrl: modelConfig.anthropic.apiBaseUrl,
    apiKey: modelConfig.anthropic.apiKey,
  })
  const geminiClient = createGeminiClient({
    dryRun: config.dryRun,
    apiBaseUrl: modelConfig.gemini.apiBaseUrl,
    apiKey: modelConfig.gemini.apiKey,
  })

  if (!config.dryRun && !anthropicClient.enabled) {
    console.error('[translation-bot] Anthropic API is required for classification and revision. Set ANTHROPIC_API_KEY.')
    process.exit(1)
  }
  if (!config.dryRun && !geminiClient.enabled) {
    console.error('[translation-bot] Gemini API is required for review. Set GEMINI_API_KEY.')
    process.exit(1)
  }

  const headSha = resolveHeadSha(args)
  const baseSha = resolveBaseSha(args, headSha)
  const candidates = getChangedSourcePosts(baseSha, headSha)

  console.log(`[translation-bot] dryRun=${config.dryRun}, base=${baseSha || 'N/A'}, head=${headSha}`)
  console.log(`[translation-bot] models: classify=${modelConfig.anthropic.classifyModel}, anth_translate=${modelConfig.anthropic.translateModel}, gem_translate=${modelConfig.gemini.translateModel}, review=${modelConfig.gemini.reviewModel}`)
  console.log(`[translation-bot] found ${candidates.length} candidate source post(s)`)

  if (candidates.length === 0)
    process.exit(0)

  const classifyPrompt = readPrompt('classify-system.txt')
  const translatePrompt = readPrompt('translate-system.txt')
  const verifyPrompt = readPrompt('verify-system.txt')

  const summary = {
    translated: [],
    skipped: [],
    errors: [],
  }

  for (const relPath of candidates) {
    try {
      const raw = readFileSync(relPath, 'utf8')
      const { frontmatter, body } = splitFrontmatter(raw)
      const meta = readMetadata(frontmatter)

      const skipReason = getSkipReason(meta)
      if (skipReason) {
        summary.skipped.push({ file: relPath, reason: skipReason })
        continue
      }

      const sourceLang = detectSourceLang({
        filePath: relPath,
        meta,
        defaultSourceLang: config.defaultSourceLang,
      })

      const targetLangs = resolveTargetLangs({
        sourceLang,
        matrix: config.targetMatrix,
        cliTargetLangs: String(args['target-langs'] || ''),
      })

      if (targetLangs.length === 0) {
        summary.skipped.push({ file: relPath, reason: `no target langs for source=${sourceLang}` })
        continue
      }

      const classification = await classifyWithAnthropic({
        dryRun: config.dryRun,
        prompt: classifyPrompt,
        frontmatter,
        body,
        meta,
        client: anthropicClient,
        model: modelConfig.anthropic.classifyModel,
      })

      const override = readTranslationOverride(meta.translation)
      const shouldTranslate = override === 'force'
        || (override !== 'skip'
          && classification.type === 'tech_share'
          && classification.confidence >= config.classificationThreshold)

      if (!shouldTranslate) {
        summary.skipped.push({
          file: relPath,
          reason: `classified=${classification.type}, confidence=${classification.confidence.toFixed(2)}`,
        })
        continue
      }

      for (const targetLang of targetLangs) {
        try {
          const targetPath = toTargetPath(relPath, targetLang)
          if (!targetPath) {
            summary.skipped.push({ file: relPath, reason: `unsupported extension ${extname(relPath)}` })
            continue
          }

          let action = 'create'
          if (existsSync(targetPath)) {
            if (!config.updateExistingTranslations) {
              summary.skipped.push({ file: relPath, reason: `target exists and updates are disabled: ${targetPath}` })
              continue
            }

            const targetRaw = readFileSync(targetPath, 'utf8')
            const { frontmatter: targetFrontmatter } = splitFrontmatter(targetRaw)
            const targetMeta = readMetadata(targetFrontmatter)
            const isManagedTranslation = targetMeta.translationGenerated === true

            if (!isManagedTranslation && !config.overwriteManualTranslations) {
              summary.skipped.push({
                file: relPath,
                reason: `target exists but is not managed translation (set OVERWRITE_MANUAL_TRANSLATIONS=true to override): ${targetPath}`,
              })
              continue
            }
            action = 'update'
          }

          const styleRefs = getStyleReferences({
            targetLang,
            categories: meta.categories,
            excludePath: relPath,
            defaultSourceLang: config.defaultSourceLang,
          })

          const candidateSet = await generateTranslationCandidates({
            dryRun: config.dryRun,
            sourceTitle: meta.title || pathStem(relPath),
            sourceLang,
            targetLang,
            sourceBody: body,
            styleRefs,
            variantsPerProvider: config.variantsPerProvider,
            translatePrompt,
            anthropicClient,
            anthropicModel: modelConfig.anthropic.translateModel,
            geminiClient,
            geminiModel: modelConfig.gemini.translateModel,
          })

          if (candidateSet.length === 0) {
            summary.errors.push({ file: relPath, error: `no translation candidates for ${targetLang}` })
            continue
          }

          let best = await selectBestByGeminiReview({
            dryRun: config.dryRun,
            verifyPrompt,
            sourceTitle: meta.title || pathStem(relPath),
            sourceLang,
            targetLang,
            sourceBody: body,
            candidates: candidateSet,
            styleRefs,
            geminiClient,
            geminiModel: modelConfig.gemini.reviewModel,
            minScore: config.verificationMinScore,
          })

          for (let attempt = 1; attempt <= config.reviewMaxRevisions; attempt += 1) {
            if (isReviewAccepted(best.review, config.verificationMinScore))
              break

            const revised = await reviseWithAnthropic({
              dryRun: config.dryRun,
              translatePrompt,
              sourceTitle: meta.title || pathStem(relPath),
              sourceLang,
              targetLang,
              sourceBody: body,
              previousTitle: best.candidate.title,
              previousBody: best.candidate.body,
              reviewFeedback: best.review,
              styleRefs,
              client: anthropicClient,
              model: modelConfig.anthropic.reviseModel,
              attempt,
            })

            const revisedReview = await reviewWithGemini({
              dryRun: config.dryRun,
              verifyPrompt,
              sourceTitle: meta.title || pathStem(relPath),
              sourceLang,
              targetLang,
              sourceBody: body,
              translatedTitle: revised.title,
              translatedBody: revised.body,
              styleRefs,
              client: geminiClient,
              model: modelConfig.gemini.reviewModel,
              minScore: config.verificationMinScore,
            })

            if (revisedReview.score >= best.review.score || revisedReview.passed) {
              best = {
                candidate: {
                  provider: 'anthropic-revise',
                  variant: attempt,
                  title: revised.title,
                  body: revised.body,
                },
                review: revisedReview,
              }
            }
          }

          if (!isReviewAccepted(best.review, config.verificationMinScore)) {
            summary.skipped.push({
              file: relPath,
              reason: `review failed for ${targetLang}: score=${best.review.score}, summary=${best.review.summary}`,
            })
            if (config.failOnError) {
              summary.errors.push({
                file: relPath,
                error: `review failed for ${targetLang}`,
              })
            }
            continue
          }

          const translatedFrontmatter = buildTranslatedFrontmatter(frontmatter, {
            title: best.candidate.title || meta.title || pathStem(relPath),
            lang: targetLang,
          })
          const output = `---\n${translatedFrontmatter}\n---\n\n${best.candidate.body.trimStart()}`
          if (!config.dryRun)
            writeFileSync(targetPath, ensureTrailingNewline(output), 'utf8')

          summary.translated.push({
            source: relPath,
            target: targetPath,
            action,
            provider: best.candidate.provider,
            score: best.review.score,
          })
        }
        catch (error) {
          summary.errors.push({ file: relPath, error: `${targetLang}: ${formatError(error)}` })
        }
      }
    }
    catch (error) {
      summary.errors.push({ file: relPath, error: formatError(error) })
    }
  }

  printSummary(summary)
  if (summary.errors.length > 0 && config.failOnError)
    process.exit(1)
}

function parseArgs(argv) {
  const args = {}
  for (let i = 0; i < argv.length; i += 1) {
    const token = argv[i]
    if (!token.startsWith('--'))
      continue
    const key = token.slice(2)
    const eqIdx = key.indexOf('=')
    if (eqIdx > -1) {
      args[key.slice(0, eqIdx)] = key.slice(eqIdx + 1)
      continue
    }
    const next = argv[i + 1]
    if (next && !next.startsWith('--')) {
      args[key] = next
      i += 1
    }
    else {
      args[key] = true
    }
  }
  return args
}

function resolveHeadSha(args) {
  const argHead = String(args.head || '').trim()
  if (argHead)
    return argHead
  return git(['rev-parse', 'HEAD'])
}

function resolveBaseSha(args, headSha) {
  const argBase = String(args.base || '').trim()
  if (argBase && !/^0+$/.test(argBase))
    return argBase
  try {
    return git(['rev-parse', `${headSha}^`])
  }
  catch {
    return ''
  }
}

function getChangedSourcePosts(baseSha, headSha) {
  if (!baseSha || !headSha)
    return []
  const output = git([
    'diff',
    '--name-status',
    '--diff-filter=AM',
    baseSha,
    headSha,
    '--',
    'content/posts',
  ], { allowEmpty: true })

  return output
    .split(/\r?\n/)
    .map(line => line.trim())
    .filter(Boolean)
    .map((line) => {
      const parts = line.split(/\s+/)
      return parts.at(-1) || ''
    })
    .filter(file => SOURCE_EXT_RE.test(file))
    .filter(file => !GENERATED_LANG_SUFFIX_RE.test(file))
}

function splitFrontmatter(markdown) {
  const match = markdown.match(/^---\r?\n([\s\S]*?)\r?\n---\r?\n?/)
  if (!match)
    return { frontmatter: '', body: markdown }
  return {
    frontmatter: match[1],
    body: markdown.slice(match[0].length),
  }
}

function readMetadata(frontmatter) {
  return {
    title: readScalar(frontmatter, 'title'),
    lang: readScalar(frontmatter, 'lang'),
    draft: readBoolean(frontmatter, 'draft'),
    password: readScalar(frontmatter, 'password'),
    translationGenerated: readBoolean(frontmatter, 'translation_generated'),
    categories: readList(frontmatter, 'categories'),
    tags: readList(frontmatter, 'tags'),
    translation: readScalar(frontmatter, 'translation'),
  }
}

function readScalar(frontmatter, key) {
  if (!frontmatter)
    return ''
  const re = new RegExp(`^${escapeRegex(key)}:\\s*(.*)$`, 'm')
  const match = frontmatter.match(re)
  if (!match)
    return ''
  return stripQuotes(match[1].trim())
}

function readBoolean(frontmatter, key) {
  const value = readScalar(frontmatter, key).toLowerCase()
  if (value === 'true')
    return true
  if (value === 'false')
    return false
  return undefined
}

function readList(frontmatter, key) {
  if (!frontmatter)
    return []
  const lines = frontmatter.split(/\r?\n/)
  const idx = lines.findIndex(line => new RegExp(`^${escapeRegex(key)}:\\s*`).test(line))
  if (idx === -1)
    return []

  const scalarMatch = lines[idx].match(new RegExp(`^${escapeRegex(key)}:\\s*(.+)$`))
  if (scalarMatch && scalarMatch[1].trim() !== '')
    return [stripQuotes(scalarMatch[1].trim())]

  const values = []
  for (let i = idx + 1; i < lines.length; i += 1) {
    const line = lines[i]
    if (!/^\s+/.test(line))
      break
    const item = line.match(/^\s*-\s*(.+?)\s*$/)
    if (item)
      values.push(stripQuotes(item[1]))
  }
  return values
}

function readTranslationOverride(value) {
  const normalized = String(value || '').trim().toLowerCase()
  if (normalized === 'skip')
    return 'skip'
  if (normalized === 'force')
    return 'force'
  return ''
}

function getSkipReason(meta) {
  if (meta.draft === true)
    return 'draft=true'
  if (meta.password)
    return 'password-protected'
  if (meta.translationGenerated === true)
    return 'translation_generated=true'
  if (readTranslationOverride(meta.translation) === 'skip')
    return 'frontmatter translation: skip'
  return ''
}

function detectSourceLang({ filePath, meta, defaultSourceLang }) {
  const fmLang = normalizeLang(meta.lang)
  if (fmLang)
    return fmLang
  const suffix = filePath.match(/\.(zh|en|ja)\.(md|mdx)$/i)
  if (suffix)
    return normalizeLang(suffix[1]) || defaultSourceLang
  return defaultSourceLang
}

function normalizeLang(value) {
  const lang = String(value || '').trim().toLowerCase()
  return SUPPORTED_LANGS.has(lang) ? lang : ''
}

function parseTargetMatrix(raw) {
  if (!raw)
    return DEFAULT_TARGET_MATRIX
  try {
    const parsed = JSON.parse(raw)
    const matrix = {}
    for (const [key, value] of Object.entries(parsed || {})) {
      const source = normalizeLang(key)
      if (!source || !Array.isArray(value))
        continue
      matrix[source] = unique(
        value
          .map(item => normalizeLang(item))
          .filter(Boolean)
          .filter(item => item !== source),
      )
    }
    return Object.keys(matrix).length > 0 ? matrix : DEFAULT_TARGET_MATRIX
  }
  catch {
    return DEFAULT_TARGET_MATRIX
  }
}

function resolveTargetLangs({ sourceLang, matrix, cliTargetLangs }) {
  if (!sourceLang)
    return []
  const cli = String(cliTargetLangs || '').trim()
  if (cli) {
    return unique(
      cli
        .split(',')
        .map(item => normalizeLang(item))
        .filter(Boolean)
        .filter(item => item !== sourceLang),
    )
  }
  return unique((matrix[sourceLang] || []).filter(item => item !== sourceLang))
}

async function classifyWithAnthropic({
  dryRun,
  prompt,
  frontmatter,
  body,
  meta,
  client,
  model,
}) {
  const heuristic = heuristicClassification({ body, meta })
  if (dryRun)
    return heuristic
  if (!client.enabled)
    return heuristic

  const userPrompt = [
    'Classify this article and return JSON only.',
    '',
    `Title: ${meta.title || '(empty)'}`,
    `Lang: ${meta.lang || '(unset)'}`,
    `Categories: ${(meta.categories || []).join(', ') || '(none)'}`,
    `Tags: ${(meta.tags || []).join(', ') || '(none)'}`,
    '',
    'Frontmatter:',
    frontmatter || '(none)',
    '',
    'Body preview (first 6000 chars):',
    body.slice(0, 6000),
  ].join('\n')

  try {
    const raw = await client.chat({
      model,
      systemPrompt: prompt,
      userPrompt,
      temperature: 0.1,
      maxTokens: 300,
    })
    const parsed = parseJsonFromText(raw)
    if (!parsed)
      return heuristic
    const type = String(parsed.type || '').trim()
    const confidence = clampNumber(Number(parsed.confidence), 0, 1)
    const rationale = String(parsed.rationale || '').trim()
    if (!CLASS_TYPES.has(type))
      return heuristic
    return {
      type,
      confidence,
      rationale: rationale || heuristic.rationale,
      source: 'anthropic',
    }
  }
  catch {
    return heuristic
  }
}

function heuristicClassification({ body, meta }) {
  const text = [
    meta.title,
    ...(meta.categories || []),
    ...(meta.tags || []),
    body.slice(0, 5000),
  ].join('\n').toLowerCase()

  const techKeywords = [
    'code', 'software', 'engineering', 'algorithm', 'api', 'llm', 'agent', 'python', 'sql', 'docker',
    'machine learning', 'deep learning', 'research', 'tutorial',
    '技术', '代码', '工程', '算法', '机器学习', '教程', '模型', '开发',
  ]
  const diaryKeywords = [
    'diary', 'my life', 'emotion', 'memory', 'travel',
    '日记', '生活', '心情', '经历', '回忆', '人生',
  ]
  const noteKeywords = ['note', 'todo', 'memo', '笔记', '记录', '备忘', '摘抄']

  const techScore = countHits(text, techKeywords)
  const diaryScore = countHits(text, diaryKeywords)
  const noteScore = countHits(text, noteKeywords)

  if (techScore >= Math.max(diaryScore + 1, noteScore + 1) && techScore >= 2) {
    return {
      type: 'tech_share',
      confidence: clampNumber(0.55 + techScore * 0.05, 0, 0.88),
      rationale: 'heuristic technical',
      source: 'heuristic',
    }
  }
  if (diaryScore > techScore && diaryScore >= noteScore) {
    return {
      type: 'diary_life',
      confidence: clampNumber(0.55 + diaryScore * 0.05, 0, 0.85),
      rationale: 'heuristic diary',
      source: 'heuristic',
    }
  }
  return {
    type: 'personal_note',
    confidence: clampNumber(0.5 + noteScore * 0.05, 0, 0.8),
    rationale: 'heuristic note',
    source: 'heuristic',
  }
}

async function generateTranslationCandidates({
  dryRun,
  sourceTitle,
  sourceLang,
  targetLang,
  sourceBody,
  styleRefs,
  variantsPerProvider,
  translatePrompt,
  anthropicClient,
  anthropicModel,
  geminiClient,
  geminiModel,
}) {
  if (dryRun) {
    return [
      {
        provider: 'dry-run',
        variant: 1,
        title: `[dry-run] ${sourceTitle}`,
        body: sourceBody,
      },
    ]
  }

  const candidates = []
  const styleContext = formatStyleRefs(styleRefs)

  if (anthropicClient.enabled) {
    for (let i = 1; i <= variantsPerProvider; i += 1) {
      const response = await translateWithProvider({
        provider: 'anthropic',
        client: anthropicClient,
        model: anthropicModel,
        prompt: translatePrompt,
        sourceTitle,
        sourceLang,
        targetLang,
        sourceBody,
        styleContext,
        variant: i,
      })
      candidates.push(response)
    }
  }

  if (geminiClient.enabled) {
    for (let i = 1; i <= variantsPerProvider; i += 1) {
      const response = await translateWithProvider({
        provider: 'gemini',
        client: geminiClient,
        model: geminiModel,
        prompt: translatePrompt,
        sourceTitle,
        sourceLang,
        targetLang,
        sourceBody,
        styleContext,
        variant: i,
      })
      candidates.push(response)
    }
  }

  return candidates.filter(item => item && item.body)
}

async function translateWithProvider({
  provider,
  client,
  model,
  prompt,
  sourceTitle,
  sourceLang,
  targetLang,
  sourceBody,
  styleContext,
  variant,
}) {
  const variantProfile = getVariantProfile(variant)
  const temperature = clampNumber(0.15 + variant * 0.15, 0.15, 0.85)
  const userPrompt = [
    `Source language: ${sourceLang}`,
    `Target language: ${targetLang}`,
    `Variant profile: ${variantProfile}`,
    '',
    'Keep terminology accurate and preserve markdown/code formatting.',
    'Match style with references when appropriate.',
    '',
    styleContext ? `Style references:\n${styleContext}` : 'Style references: (none)',
    '',
    `Title: ${sourceTitle}`,
    '',
    'Markdown body:',
    sourceBody,
  ].join('\n')

  const raw = await client.chat({
    model,
    systemPrompt: prompt,
    userPrompt,
    temperature,
    maxTokens: 7000,
  })
  const parsed = parseJsonFromText(raw)
  if (!parsed)
    throw new Error(`${provider} variant ${variant} returned invalid JSON`)
  const title = String(parsed.title || '').trim()
  const body = String(parsed.body_markdown || '').trim()
  if (!body)
    throw new Error(`${provider} variant ${variant} missing body_markdown`)
  return {
    provider,
    variant,
    title,
    body,
  }
}

async function selectBestByGeminiReview({
  dryRun,
  verifyPrompt,
  sourceTitle,
  sourceLang,
  targetLang,
  sourceBody,
  candidates,
  styleRefs,
  geminiClient,
  geminiModel,
  minScore,
}) {
  if (dryRun) {
    return {
      candidate: candidates[0],
      review: {
        score: 100,
        passed: true,
        summary: 'dry run',
        majorIssues: [],
        minorIssues: [],
      },
    }
  }

  const reviewed = []
  for (const candidate of candidates) {
    const review = await reviewWithGemini({
      dryRun,
      verifyPrompt,
      sourceTitle,
      sourceLang,
      targetLang,
      sourceBody,
      translatedTitle: candidate.title,
      translatedBody: candidate.body,
      styleRefs,
      client: geminiClient,
      model: geminiModel,
      minScore,
    })
    reviewed.push({ candidate, review })
  }

  reviewed.sort((a, b) => {
    if (b.review.score !== a.review.score)
      return b.review.score - a.review.score
    if (Number(b.review.passed) !== Number(a.review.passed))
      return Number(b.review.passed) - Number(a.review.passed)
    return 0
  })
  return reviewed[0]
}

async function reviewWithGemini({
  dryRun,
  verifyPrompt,
  sourceTitle,
  sourceLang,
  targetLang,
  sourceBody,
  translatedTitle,
  translatedBody,
  styleRefs,
  client,
  model,
  minScore,
}) {
  if (dryRun) {
    return {
      score: 100,
      passed: true,
      summary: 'dry run',
      majorIssues: [],
      minorIssues: [],
    }
  }

  const styleContext = formatStyleRefs(styleRefs)
  const userPrompt = [
    `Source title: ${sourceTitle}`,
    `Source language: ${sourceLang}`,
    `Target language: ${targetLang}`,
    `Passing score threshold: ${minScore}`,
    '',
    styleContext ? `Reference style samples in ${targetLang}:\n${styleContext}` : 'Reference style samples: (none)',
    '',
    'Source markdown:',
    sourceBody.slice(0, 12000),
    '',
    'Translated title:',
    translatedTitle || '(empty)',
    '',
    'Translated markdown:',
    translatedBody.slice(0, 12000),
  ].join('\n')

  const raw = await client.chat({
    model,
    systemPrompt: verifyPrompt,
    userPrompt,
    temperature: 0,
    maxTokens: 800,
  })
  const parsed = parseJsonFromText(raw)
  if (!parsed) {
    return {
      score: 0,
      passed: false,
      summary: 'review response is not valid JSON',
      majorIssues: ['invalid review JSON'],
      minorIssues: [],
    }
  }

  const score = clampNumber(Number(parsed.score), 0, 100)
  const summary = String(parsed.summary || '').trim()
  const majorIssues = Array.isArray(parsed.major_issues) ? parsed.major_issues.map(item => String(item)) : []
  const minorIssues = Array.isArray(parsed.minor_issues) ? parsed.minor_issues.map(item => String(item)) : []
  const passed = Boolean(parsed.passed) && score >= minScore && majorIssues.length === 0

  return {
    score,
    passed,
    summary,
    majorIssues,
    minorIssues,
  }
}

async function reviseWithAnthropic({
  dryRun,
  translatePrompt,
  sourceTitle,
  sourceLang,
  targetLang,
  sourceBody,
  previousTitle,
  previousBody,
  reviewFeedback,
  styleRefs,
  client,
  model,
  attempt,
}) {
  if (dryRun) {
    return {
      title: previousTitle || `[dry-run revised] ${sourceTitle}`,
      body: previousBody,
    }
  }

  const styleContext = formatStyleRefs(styleRefs)
  const userPrompt = [
    `Source language: ${sourceLang}`,
    `Target language: ${targetLang}`,
    `Revision attempt: ${attempt}`,
    '',
    'Please revise the translation according to review feedback.',
    '',
    'Review summary:',
    reviewFeedback.summary || '(none)',
    '',
    `Major issues: ${(reviewFeedback.majorIssues || []).join('; ') || '(none)'}`,
    `Minor issues: ${(reviewFeedback.minorIssues || []).join('; ') || '(none)'}`,
    '',
    styleContext ? `Reference style samples:\n${styleContext}` : 'Reference style samples: (none)',
    '',
    `Original title: ${sourceTitle}`,
    'Original markdown:',
    sourceBody,
    '',
    `Previous translated title: ${previousTitle || '(empty)'}`,
    'Previous translated markdown:',
    previousBody,
  ].join('\n')

  const raw = await client.chat({
    model,
    systemPrompt: translatePrompt,
    userPrompt,
    temperature: 0.2,
    maxTokens: 7000,
  })
  const parsed = parseJsonFromText(raw)
  if (!parsed)
    throw new Error('revision response is not valid JSON')
  const title = String(parsed.title || '').trim()
  const body = String(parsed.body_markdown || '').trim()
  if (!body)
    throw new Error('revision response missing body_markdown')
  return { title, body }
}

function isReviewAccepted(review, minScore) {
  if (!review)
    return false
  if (review.passed !== true)
    return false
  if (review.score < minScore)
    return false
  if ((review.majorIssues || []).length > 0)
    return false
  return true
}

function getVariantProfile(index) {
  const profiles = [
    'faithful technical wording, minimal stylistic changes',
    'concise explanatory blog tone, keep technical precision',
    'reader-friendly narrative while preserving meaning',
    'terminology-first translation, strict consistency',
  ]
  return profiles[(index - 1) % profiles.length]
}

function toTargetPath(sourcePath, targetLang) {
  if (!SOURCE_EXT_RE.test(sourcePath))
    return ''
  const base = sourcePath.replace(/\.(zh|en|ja)(?=\.(md|mdx)$)/i, '')
  return base.replace(/(\.md|\.mdx)$/i, `.${targetLang}$1`)
}

function pathStem(filePath) {
  return filePath.split('/').at(-1)?.replace(/\.(md|mdx)$/i, '') || filePath
}

function loadStyleCorpus(defaultSourceLang) {
  if (styleCorpusCache)
    return styleCorpusCache
  const files = git(['ls-files', 'content/posts'], { allowEmpty: true })
    .split(/\r?\n/)
    .filter(Boolean)
    .filter(file => SOURCE_EXT_RE.test(file))

  const corpus = []
  for (const file of files) {
    try {
      const raw = readFileSync(file, 'utf8')
      const { frontmatter, body } = splitFrontmatter(raw)
      const meta = readMetadata(frontmatter)
      if (meta.draft === true || meta.password)
        continue
      const lang = detectSourceLang({
        filePath: file,
        meta,
        defaultSourceLang,
      })
      if (!lang)
        continue
      const excerpt = compactText(body).slice(0, 500)
      if (excerpt.length < 80)
        continue
      corpus.push({
        path: file,
        lang,
        title: meta.title || pathStem(file),
        categories: meta.categories || [],
        excerpt,
      })
    }
    catch {
      // ignore broken reference files
    }
  }
  styleCorpusCache = corpus
  return styleCorpusCache
}

function getStyleReferences({ targetLang, categories, excludePath, defaultSourceLang }) {
  const corpus = loadStyleCorpus(defaultSourceLang)
  const srcCategories = new Set((categories || []).map(item => String(item).toLowerCase()))

  const scored = corpus
    .filter(item => item.lang === targetLang)
    .filter(item => item.path !== excludePath)
    .map((item) => {
      const overlap = item.categories.reduce((acc, c) => {
        return acc + (srcCategories.has(String(c).toLowerCase()) ? 1 : 0)
      }, 0)
      const suffixBonus = item.path.includes(`.${targetLang}.`) ? 1 : 0
      return {
        item,
        score: overlap * 10 + suffixBonus,
      }
    })
    .sort((a, b) => b.score - a.score)
    .slice(0, 3)
    .map(x => x.item)

  return scored
}

function formatStyleRefs(refs) {
  if (!refs || refs.length === 0)
    return ''
  return refs.map((ref, idx) => {
    const cats = (ref.categories || []).join(', ') || '(none)'
    return [
      `[Ref ${idx + 1}]`,
      `Title: ${ref.title}`,
      `Categories: ${cats}`,
      `Excerpt: ${ref.excerpt}`,
    ].join('\n')
  }).join('\n\n')
}

function buildTranslatedFrontmatter(frontmatter, updates) {
  const lines = (frontmatter || '')
    .split(/\r?\n/)
    .filter((line, idx, arr) => !(idx === arr.length - 1 && line.trim() === ''))

  const next = [...lines]
  setOrInsertScalar(next, 'title', toYamlScalar(updates.title))
  setOrInsertScalar(next, 'lang', toYamlScalar(updates.lang))
  setOrInsertScalar(next, 'translation_generated', 'true')
  return next.join('\n').trimEnd()
}

function setOrInsertScalar(lines, key, serializedValue) {
  const keyRe = new RegExp(`^${escapeRegex(key)}:\\s*`)
  const idx = lines.findIndex(line => keyRe.test(line))
  const newLine = `${key}: ${serializedValue}`
  if (idx >= 0) {
    lines[idx] = newLine
    return
  }
  if (key === 'lang') {
    const titleIdx = lines.findIndex(line => /^title:\s*/.test(line))
    if (titleIdx >= 0) {
      lines.splice(titleIdx + 1, 0, newLine)
      return
    }
  }
  lines.push(newLine)
}

function createAnthropicClient({ dryRun, apiBaseUrl, apiKey }) {
  if (dryRun) {
    return {
      enabled: true,
      chat: async () => JSON.stringify({}),
    }
  }
  return {
    enabled: Boolean(apiKey),
    chat: async ({ model, systemPrompt, userPrompt, temperature, maxTokens }) => {
      const response = await fetch(apiBaseUrl, {
        method: 'POST',
        headers: {
          'content-type': 'application/json',
          'x-api-key': apiKey,
          'anthropic-version': '2023-06-01',
        },
        body: JSON.stringify({
          model,
          system: systemPrompt,
          max_tokens: maxTokens,
          temperature,
          messages: [{ role: 'user', content: userPrompt }],
        }),
      })
      const payload = await response.json().catch(() => ({}))
      if (!response.ok)
        throw new Error(`anthropic request failed: ${response.status} ${JSON.stringify(payload)}`)
      const text = payload?.content?.find?.(part => part.type === 'text')?.text || ''
      return String(text).trim()
    },
  }
}

function createGeminiClient({ dryRun, apiBaseUrl, apiKey }) {
  if (dryRun) {
    return {
      enabled: true,
      chat: async () => JSON.stringify({}),
    }
  }
  return {
    enabled: Boolean(apiKey),
    chat: async ({ model, systemPrompt, userPrompt, temperature, maxTokens }) => {
      const endpoint = buildGeminiEndpoint(apiBaseUrl, model)
      const response = await fetch(endpoint, {
        method: 'POST',
        headers: {
          'content-type': 'application/json',
          'x-goog-api-key': apiKey,
        },
        body: JSON.stringify({
          systemInstruction: {
            parts: [{ text: systemPrompt }],
          },
          contents: [
            {
              role: 'user',
              parts: [{ text: userPrompt }],
            },
          ],
          generationConfig: {
            temperature,
            maxOutputTokens: maxTokens,
          },
        }),
      })
      const payload = await response.json().catch(() => ({}))
      if (!response.ok)
        throw new Error(`gemini request failed: ${response.status} ${JSON.stringify(payload)}`)
      const text = payload?.candidates?.[0]?.content?.parts?.map(part => part?.text || '').join('') || ''
      return String(text).trim()
    },
  }
}

function buildGeminiEndpoint(baseUrl, model) {
  const normalized = String(baseUrl || '').trim().replace(/\/+$/, '')
  if (!normalized) {
    return `https://generativelanguage.googleapis.com/v1beta/models/${encodeURIComponent(model)}:generateContent`
  }
  if (normalized.includes('{model}'))
    return normalized.replace('{model}', encodeURIComponent(model))
  if (normalized.includes(':generateContent'))
    return normalized
  if (normalized.endsWith('/v1beta') || normalized.endsWith('/v1'))
    return `${normalized}/models/${encodeURIComponent(model)}:generateContent`
  if (normalized.endsWith('/models'))
    return `${normalized}/${encodeURIComponent(model)}:generateContent`
  return normalized
}

function readPrompt(fileName) {
  return readFileSync(join(__dirname, 'prompts', fileName), 'utf8').trim()
}

function git(args, { allowEmpty = false } = {}) {
  try {
    return execFileSync('git', args, { encoding: 'utf8' }).trim()
  }
  catch (error) {
    if (allowEmpty)
      return ''
    throw error
  }
}

function getEnv(name) {
  const value = process.env[name]
  return typeof value === 'string' ? value.trim() : ''
}

function parseJsonFromText(text) {
  if (!text)
    return null
  const trimmed = text.trim()
  try {
    return JSON.parse(trimmed)
  }
  catch {
    const start = trimmed.indexOf('{')
    const end = trimmed.lastIndexOf('}')
    if (start === -1 || end === -1 || end <= start)
      return null
    try {
      return JSON.parse(trimmed.slice(start, end + 1))
    }
    catch {
      return null
    }
  }
}

function stripQuotes(value) {
  const text = String(value || '').trim()
  if ((text.startsWith('"') && text.endsWith('"')) || (text.startsWith('\'') && text.endsWith('\'')))
    return text.slice(1, -1).trim()
  return text
}

function compactText(text) {
  return String(text || '').replace(/\s+/g, ' ').trim()
}

function escapeRegex(text) {
  return text.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')
}

function countHits(text, keywords) {
  return keywords.reduce((acc, keyword) => (text.includes(keyword) ? acc + 1 : acc), 0)
}

function unique(items) {
  return Array.from(new Set(items))
}

function toBoolean(value) {
  const normalized = String(value || '').trim().toLowerCase()
  return ['1', 'true', 'yes', 'on'].includes(normalized)
}

function isExplicitFalse(value) {
  const normalized = String(value || '').trim().toLowerCase()
  return ['0', 'false', 'no', 'off'].includes(normalized)
}

function clampNumber(value, min, max) {
  if (!Number.isFinite(value))
    return min
  return Math.min(Math.max(value, min), max)
}

function clampInt(value, min, max) {
  if (!Number.isFinite(value))
    return min
  return Math.min(Math.max(Math.round(value), min), max)
}

function toYamlScalar(value) {
  const text = String(value || '').trim()
  if (text === '')
    return '\'\''
  const simple = /^[\p{L}\p{N}\-_. /]+$/u.test(text)
  const reserved = /^(true|false|null|~|yes|no|on|off|-?\d+(\.\d+)?)$/i.test(text)
  if (simple && !reserved)
    return text
  const escaped = text.replace(/\\/g, '\\\\').replace(/"/g, '\\"')
  return `"${escaped}"`
}

function ensureTrailingNewline(text) {
  return text.endsWith('\n') ? text : `${text}\n`
}

function printSummary(summary) {
  for (const item of summary.translated) {
    console.log(`[translation-bot] translated: ${item.source} -> ${item.target} (action=${item.action}, provider=${item.provider}, score=${item.score})`)
  }
  for (const item of summary.skipped) {
    console.log(`[translation-bot] skipped: ${item.file} (${item.reason})`)
  }
  for (const item of summary.errors) {
    console.log(`[translation-bot] error: ${item.file} (${item.error})`)
  }
  console.log(`[translation-bot] done. translated=${summary.translated.length}, skipped=${summary.skipped.length}, errors=${summary.errors.length}`)
}

function formatError(error) {
  if (error instanceof Error)
    return `${error.name}: ${error.message}`
  return String(error)
}

main().catch((error) => {
  console.error(`[translation-bot] fatal: ${formatError(error)}`)
  process.exit(1)
})
