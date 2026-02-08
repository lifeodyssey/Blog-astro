# Translation Bot (Blog-astro)

This folder contains the translation pipeline used by `.github/workflows/translate-new-posts.yml`.

## What it does

1. Detects changed source files in `content/posts` (new or modified).
2. Classifies each new post into:
   - `tech_share`
   - `personal_note`
   - `diary_life`
3. Translates only `tech_share` posts with confidence above threshold.
4. Runs a verification bot to score translation quality.
5. Writes translated files and opens an automated PR for review.

Default language targets:

- `zh` source -> `en`, `ja`
- `ja` source -> `zh`, `en`
- `en` source -> `zh`, `ja`

You can override with `TRANSLATION_TARGET_MATRIX_JSON`.

## Frontmatter controls

- `translation: skip` -> always skip this post.
- `translation: force` -> always translate this post.

## Model config (Secrets only)

All URL/API key/model values are read from GitHub Secrets.

Anthropic secrets:

- `ANTHROPIC_API_BASE_URL` (optional; default official endpoint)
- `ANTHROPIC_API_KEY` (required)
- `ANTHROPIC_MODEL` (optional global default)
- `ANTHROPIC_MODEL_CLASSIFY` (optional)
- `ANTHROPIC_MODEL_TRANSLATE` (optional)
- `ANTHROPIC_MODEL_REVISE` (optional)

Gemini secrets:

- `GEMINI_API_BASE_URL` (optional; supports custom gateway URL)
- `GEMINI_API_KEY` (required)
- `GEMINI_MODEL` (optional global default)
- `GEMINI_MODEL_TRANSLATE` (optional)
- `GEMINI_MODEL_REVIEW` (optional)

Other behavior secrets:

- `DEFAULT_SOURCE_LANG` (default `zh`)
- `TRANSLATION_TARGET_MATRIX_JSON` (default `{"zh":["en","ja"],"ja":["zh","en"],"en":["zh","ja"]}`)
- `CLASSIFICATION_CONFIDENCE_THRESHOLD` (default `0.75`)
- `VERIFICATION_MIN_SCORE` (default `80`)
- `VARIANTS_PER_PROVIDER` (default `2`, total candidates = `2 * providers`)
- `REVIEW_MAX_REVISIONS` (default `1`)
- `TRANSLATION_FAIL_ON_ERROR` (default `false`)
- `UPDATE_EXISTING_TRANSLATIONS` (default `true`)
- `OVERWRITE_MANUAL_TRANSLATIONS` (default `false`, only overwrite files with `translation_generated: true` unless enabled)

Review and rollback behavior:

- Gemini reviews all candidates and selects the best one.
- If review does not pass threshold, Anthropic revises using feedback.
- Revised version is re-reviewed by Gemini.
- If still failing after max revisions, output is skipped (or fails job if `TRANSLATION_FAIL_ON_ERROR=true`).

## Local dry run

```bash
node scripts/translation/run.mjs --base HEAD~1 --head HEAD --dry-run true
```
