---
title: javascript
tags:
  - Javascript 
  - 'Software Engineering'
categories: 学习笔记
abbrlink: 84ea90f7
password: GTB2o22
date: 2022-06-27 15:45:10
mathjax:
copyright:
---

后面还要学typescirpt

<!-- more -->

# Basic

js会自动进行强制类型转换 所以3/4=0.75

```javascript
    const actual = [
      /* eslint-disable no-self-compare, eqeqeq, yoda */
      1 == 1,
      '1' == 1,
      1 == '1',
      0 == false,
      0 == null,
      objectLeft == objectRight,
      0 == undefined,
      null == undefined,
      /* eslint-enable */
    ];
    const expected = [true, true, true, 
                      true, false, false, false, true];
```

在进行比较的时候会自动转换成同一类型

```javascript
    //should not perform type conversion for strict equal operator
    const actual = [
      /* eslint-disable no-self-compare, eqeqeq, yoda */
      3 === 3,
      3 === '3',
      objectLeft === objectRight,
      null === undefined,
      /* eslint-enable */
    ];
    const expected = [true, false, false, false];
```

strict equal则不会

```javascript
    let i = 1000;
    // eslint-disable-next-line no-empty, no-shadow
    for (let i = 0; i <= 5; i += 1) {}

    // <--start
    // Please write down the correct value. You should write the final result directly.
    const expected = 1000;
    
        for (var i = 0; i <= 5; i += 1) {}

    // <--start
    // Please write down the correct value. You should write the final result directly.
    const expected = 6;
```

var 声明是全局作用域或函数作用域，而 let 和 const 是块作用域。 var 变量可以在其范围内更新和重新声明； let 变量可以被更新但不能重新声明； const 变量既不能更新也不能重新声明。 它们都被提升到其作用域的顶端

push element into array用的是array.push(6, 7, 8);

```javascript
    const numbers = [1, 2, 3, 4, 5];
    const mapped = numbers.map((n, i) => `Unit ${n} for element at index ${i}`);

    // <--start
    // Please write down the correct value. You should write the final result directly.
    const expected = [
      'Unit 1 for element at index 0',
      'Unit 2 for element at index 1',
      'Unit 3 for element at index 2',
      'Unit 4 for element at index 3',
      'Unit 5 for element at index 4',
    ];
```

`**map()**` 方法创建一个新数组，这个新数组由原数组中的每个元素都调用一次提供的函数后的返回值组成。

let mappedArr = arr.map(function callback(currentValue[, index[, array]]) { // Return element for new_array  }[, thisArg])

function有(function callback(currentValue[, index[, array]])可以接受三个参数 分别是当前的正在处理的数组元素、当前处理元素的索引、调用map方法的数组

**`reduce()`** 方法对数组中的每个元素按序执行一个由您提供的 **reducer** 函数，每一次运行 **reducer** 会将先前元素的计算结果作为参数传入，最后将其结果汇总为单个返回值。

第一次执行回调函数时，不存在“上一次的计算结果”。如果需要回调函数从数组索引为 0 的元素开始执行，则需要传递初始值。否则，数组索引为 0 的元素将被作为初始值 *initialValue*，迭代器将从第二个元素开始执行（索引为 1 而不是 0）。

参数

```javascript
// Arrow function
reduce((previousValue, currentValue) => { /* ... */ } )
reduce((previousValue, currentValue, currentIndex) => { /* ... */ } )
reduce((previousValue, currentValue, currentIndex, array) => { /* ... */ } )
reduce((previousValue, currentValue, currentIndex, array) => { /* ... */ }, initialValue)

// Callback function
reduce(callbackFn)
reduce(callbackFn, initialValue)

// Inline callback function
reduce(function(previousValue, currentValue) { /* ... */ })
reduce(function(previousValue, currentValue, currentIndex) { /* ... */ })
reduce(function(previousValue, currentValue, currentIndex, array) { /* ... */ })
reduce(function(previousValue, currentValue, currentIndex, array) { /* ... */ }, initialValue)
```

- `callbackFn`

  一个 “reducer” 函数，包含四个参数：`previousValue`：上一次调用 `callbackFn` 时的返回值。在第一次调用时，若指定了初始值 `initialValue`，其值则为 `initialValue`，否则为数组索引为 0 的元素 `array[0]`。`currentValue`：数组中正在处理的元素。在第一次调用时，若指定了初始值 `initialValue`，其值则为数组索引为 0 的元素 `array[0]`，否则为 `array[1]`。`currentIndex`：数组中正在处理的元素的索引。若指定了初始值 `initialValue`，则起始索引号为 0，否则从索引 1 起始。`array`：用于遍历的数组。

- `initialValue` 可选

  作为第一次调用 `callback` 函数时参数 *previousValue* 的值。若指定了初始值 `initialValue`，则 `currentValue` 则将使用数组第一个元素；否则 `previousValue` 将使用数组第一个元素，而 `currentValue` 将使用数组第二个元素。

返回值是使用 “reducer” 回调函数遍历整个数组后的结果。

可选参数https://www.bookstack.cn/read/eloquent-js-3e-zh/3.8.md



JavaScript 对传入函数的参数数量几乎不做任何限制。如果你传递了过多参数，多余的参数就会被忽略掉，而如果你传递的参数过少，遗漏的参数将会被赋值成`undefined`。

该特性的缺点是你可能恰好向函数传递了错误数量的参数，但没有人会告诉你这个错误。

```javascript

'should pass pre-defined function as callback'
function repeat(n, action) {
  for (let i = 0; i < n; i += 1) {
    action(i);
  }
}
const labels = [];
repeat(3, (index) => labels.push(index * 3));

// <--start
// Please write down the correct value. You should write the final result directly.
const expected = [0, 3, 6];
// --end->
```

这个是传入了两个参数 一个是n，一个是函数，看懂了

```javascript
  it('should not make you crazy even we change the control flow', () => {
    function unless(test, then) {
      if (!test) then();
    }

    function repeat(n, action) {
      for (let i = 0; i < n; i += 1) {
        action(i);
      }
    }

    const logs = [];

    repeat(5, (n) => {
      unless(n % 2 === 1, () => logs.push(n));
    });

    // <--start
    // Please write down the correct value. You should write the final result directly.
    const expected = [0,2,4];
    // --end->
```

慢慢看就看懂了，传入的一个是5，一个是一个函数

(n) => {
      unless(n % 2 === 1, () => logs.push(n));
    }

这个函数会把n传入到unless里

unless(n % 2 === 1, () => logs.push(n))

n % 2 === 1是test,如果!test就执行then 即() => logs.push(n)

当取数的位置超出长度时，返回的是undefined

# Advance

```javascript
test('统计所有类型的数量', () => {
  const types = {
    A: '3',
    B: '4',
    C: '5',
  };

  const result = countTypesNumber(types);
  expect(result).toBe(12);
  
  
  export default function countTypesNumber(source) {
  // TODO: 在这里写实现代码
    return Object.values(source).reduce((acc, item) => acc + Number(item), 0);
}
```

`**Object.values()**`方法返回一个给定对象自身的所有可枚举属性值的数组，值的顺序与使用[`for...in`](https://developer.mozilla.org/zh-CN/docs/Web/JavaScript/Reference/Statements/for...in)循环的顺序相同 ( 区别在于 for-in 循环枚举原型链中的属性 )。



```javascript
var obj = { foo: 'bar', baz: 42 };
console.log(Object.values(obj)); // ['bar', 42]

// array like object
var obj = { 0: 'a', 1: 'b', 2: 'c' };
console.log(Object.values(obj)); // ['a', 'b', 'c']

// array like object with random key ordering
// when we use numeric keys, the value returned in a numerical order according to the keys
var an_obj = { 100: 'a', 2: 'b', 7: 'c' };
console.log(Object.values(an_obj)); // ['b', 'c', 'a']

// getFoo is property which isn't enumerable
var my_obj = Object.create({}, { getFoo: { value: function() { return this.foo; } } });
my_obj.foo = 'bar';
console.log(Object.values(my_obj)); // ['bar']

// non-object argument will be coerced to an object
console.log(Object.values('foo')); // ['f', 'o', 'o']
```

```javascript

test('合并源对象并添加新的编号', () => {
  const source = {
    type: 'A',
    properties: {
      color: 'green',
      status: 'raw',
    },
  };
  const expected = {
    serialNumber: '12345',
    type: 'B',
    properties: {
      color: 'green',
      status: 'raw',
    },
  };
      const result = addSerialNumber(source);
  expect(result).toEqual(expected);
    
    export default function addSerialNumber(source) {
  // TODO: 在这里写实现代码
  return { ...source, type: 'B', serialNumber: '12345' };
}
```

如果两个对象都有一个具有相同名称的属性，则第二个对象属性将覆盖第一个对象。，所以写在后面。

关于this

```javascript
describe('for this', () => {
  test('default behavior', () => {
    const obj = {
      foo() {
        return this;
      },
      bar: 1,
    };

    const foo2 = obj.foo;

    // <--start
    // TODO: Please write down the correct value. You should write the final result directly.
    const expectedObjFoo = obj;
    const expectedFoo2 = undefined;
      /*
      this的值是动态的，自动绑定上一级。
      什么意思呢，比如obj.foo，绑定的是obj，
      但是如果直接把obj.foo拿出来即foo2，foo2其实等于
      foo(){
      foo()
      return this;
      }
      并没有上一级了
      所以此时是undefined
      */
    // --end->

    expect(obj.foo()).toEqual(expectedObjFoo);
    expect(foo2()).toEqual(expectedFoo2);
  });

  test('bind this', () => {
    const obj = {
      foo() {
        return this;
      },
      bar: 1,
    };

    const foo2 = obj.foo;
    const foo3 = foo2.bind(obj);

    // <--start
    // TODO: Please write down the correct value. You should write the final result directly.
    const expectedObjFoo = obj;
    const expectedFoo2 = undefined;
    const expectedFoo3 = obj;
    // --end->

    expect(obj.foo()).toEqual(expectedObjFoo);
    expect(foo2()).toEqual(expectedFoo2);
    expect(foo3()).toEqual(expectedFoo3);
      //bind可以强行绑定this\
      //https://segmentfault.com/a/1190000011194676
  });
});
```

关于类，我就直接把那个作业写上来了

## 关于类的作业

写一个 Person 类，要有 name，age 属性，要有一个 introduce 方法，
introduce 方法返回一个字符串形如：

> My name is Tom. I am 21 years old.

再写一个 Student 类继承 Person 类，除了 name，age 属性，还有要有 class 属性。也有一个 introduce 方法，
introduce 方法返回一个字符串形如：

> My name is Tom. I am 21 years old. I am a Student. I am at Class 2.

再写一个 Worker 类继承 Person 类，只有 name，age 属性。也有一个 introduce 方法，
introduce 方法返回一个字符串形如：

> My name is Tom. I am 21 years old. I am a Worker. I have a job.

Student 和 Worker 的这段文字：

> My name is Tom. I am 21 years old.

应该调用 Person 的一个方法 introduce 来返回。

调用的index.test.js是这么写的

```javascript
import Person from './person';
import Student from './student';
import Worker from './worker';

describe('Person', () => {
  test('should have field name and age', () => {
    const person = new Person('Tom', 21);
    expect(person.name).toBe('Tom');
    expect(person.age).toBe(21);
  });

  test('should have a method introduce, introduce person with name and age', () => {
    const person = new Person('Tom', 21);
    const introduce = person.introduce();
    expect(introduce).toBe('My name is Tom. I am 21 years old.');
  });
});

describe('Student', () => {
  test('should have field name, age and class number', () => {
    const student = new Student('Tom', 21, 2);
    expect(student.name).toBe('Tom');
    expect(student.age).toBe(21);
    expect(student.klass).toBe(2);
  });

  test('should overwrite Person introduce, introduce with name, age and class number', () => {
    const student = new Student('Tom', 21, 2);
    const introduce = student.introduce();
    expect(introduce).toBe(
      'My name is Tom. I am 21 years old. I am a Student. I am at Class 2.'
    );
  });
});

describe('Worker', () => {
  test('should have field name, age', () => {
    const worker = new Worker('Tom', 21);
    expect(worker.name).toBe('Tom');
    expect(worker.age).toBe(21);
  });

  test('should overwrite Person introduce, introduce with name and age, but different with Person introduce', () => {
    const worker = new Worker('Tom', 21);
    const introduce = worker.introduce();
    expect(introduce).toBe(
      'My name is Tom. I am 21 years old. I am a Worker. I have a job.'
    );
  });
});

```

person.js

```javascript
// TODO: 在这里写实现代码
class Person {
    constructor(name, age) {
        this.name = name;
        this.age = age;
    }

    introduce() {
        return `My name is ${this.name}. I am ${this.age} years old.`;
    }
}

export default Person;

```

student.js

```javascript
// TODO: 在这里写实现代码

import Person from './person';

class Student extends Person {
    constructor(name, age, klass) {
        super(name, age);
        this.klass = klass;
    }

    introduce() {
        return `${super.introduce()} I am a Student. I am at Class ${this.klass}.`;
    }
}

export default Student;
```

worker.js

```javascript
// TODO: 在这里写实现代码
import Person from './person';

class Worker extends Person {
    introduce() {
        return `${super.introduce()} I am a Worker. I have a job.`;
    }
}

export default Worker;
//TODO need to remark
```

## Async

这部分暂时看不懂，但是感觉用不到，等用到了再回来看。

# Bronze

三个题目 看不懂 直接抄吧

```javascript
/**
 * This function creates an array of unique values, in order, from all given
 * arrays. The result array should contains all the values of the input arrays.
 *
 * Please note that this function should correctly handle `undefined`, `null`
 * and `NaN`. Two `undefined` values should be considered as equal, two
 * `NaN` values should be considered as equal though `NaN !== NaN` and so does
 * `null`.
 *
 * @param  {...any} arrays The input arrays.
 */
export default function union(...arrays) {
  // TODO: Please implement the function
  // <-start-
  return [...new Set(arrays.flatMap((array) => array))];//
  /*
  var arr1 = [1, 2, 3, 4];

arr1.map(x => [x * 2]);
// [[2], [4], [6], [8]]

arr1.flatMap(x => [x * 2]);
// [2, 4, 6, 8]

// only one level is flattened
arr1.flatMap(x => [[x * 2]]);
// [[2], [4], [6], [8]]
set去除重复值
   */

  // --end-->
}

// TODO
// You can add additional method if you want
// <-start-

// --end-->

```

```javascript
/**
 * This class is used to calculate the total price of the products bought.
 *
 * Each products contains an `id`, which will be printed on the box. The
 * `RecepitCalculator` will scan the ids of the products and give the total price
 * of the selected products.
 */
class RecepitCalculator {
  constructor(products) {
    this.products = [...products];
  }

  /**
   * Calculate total price according to the selected product ids. Please note that
   * the `id` list can contain duplicated ids (a product can be bought multiple times).
   *
   * @param {string[]} selectProductIds An array of selected product ids. Please note
   *   that this array will contains at most 3000 items.
   * @returns The total price of the selected products.
   * @throws the `selectProductIds` is `null` or `undefined`.
   */
  getTotalPrice(selectProductIds) {
    // TODO: Please implement the method
    // <-start-
    if (!selectProductIds) {
      throw Error('Please provide selected product ids.');
    }
    return selectProductIds
        .map((id) => this.getProduct(id))
        .map((p) => p.price)
        .reduce((x, y) => x + y, 0);
    // --end->
  }

  // TODO:
  //
  // You can add addtional helper functions if you want
  // <-start-
  // --end-->
}

export default RecepitCalculator;

```

```javascript
/**
 * Return a string represents the multiply table. The returned multiply table should match
 * the following rules:
 *
 * - The table should begin with `start * start`.
 * - The first number in each expression should be increased by 1 per row.
 * - The second number in each expression should be increased by 1 per column.
 * - Each column should be left aligned (filled by ' ').
 * - Each column width should be equals to the maximum width of the expression in that column plus 2.
 * - The line break character should be `'\n'`, The whitespace character should be `' '`.
 *
 * For example, suppose that the `start` is 2 and `end` is 4. The output should be
 *
 * ```
 * 2*2=4
 * 3*2=6  3*3=9
 * 4*2=8  4*3=12  4*4=16
 * ```
 *
 * Take another example. Suppose that the `start` is 2 and `end` is 5. The output should be
 *
 * ```
 * 2*2=4
 * 3*2=6   3*3=9
 * 4*2=8   4*3=12  4*4=16
 * 5*2=10  5*3=15  5*4=20  5*5=25
 * ```
 * @param options An object containing the `start`(inclusive) and `end`(inclusive) of the multiply table.
 * @return A string which represents the multiply table.
 *
 * @throws The `options` is `null` or `undefined`. Or `start` or `end` is not provided.
 * @throws The `options.start` or `options.end` is less than or equal to zero.
 * @throws The `options.start` or `options.end` is greater than 3000.
 * @throws The `options.start` is greater than `options.end`.
 */
export default function createMultiplyTable(options) {
  // TODO: Please implement this function.
  // <--start--
  if (
    !options ||
    !options.start ||
    !options.end ||
    options.start <= 0 ||
    options.start > options.end ||
    options.end > 3000
  ) {
    throw new Error();
  }
  let result = '';
  for (let row = options.start; row <= options.end; ++row) {
    for (let col = options.start; col <= row; ++col) {
      result += getAlignedFormula(row, col, options.end);
    }
    result += '\n';
  }
  return result;

  // --end-->
}

// TODO: You can add additional functions if needed.
// <--start--
function getFormula(row, col) {
  return `${row}*${col}=${row * col}`;
}

function getCurAlignLength(col, maxRow) {
  return getFormula(maxRow, col).length + 2;
}

function getAlignedFormula(row, col, maxRow) {
  return getFormula(row, col).padEnd(getCurAlignLength(col, maxRow));
}
// --end-->
```

# dom and event

