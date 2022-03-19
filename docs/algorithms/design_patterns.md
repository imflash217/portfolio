<!-- ---
hide:
  - navigation # Hide navigation
  - toc        # Hide table of contents
--- -->

# Design Patterns for Humans 🤗

**Design Patterns** are guidelines to recurring problems; {==guidelines on how to solve certain problems==}
They are not classes, packages, or libraries that you can plug into an application and wait for magic to happen.
These are rather guideline son how to solve certain problems in certain situations.

> Design Patterns are guidelines to solving certain recurring problems.

Wikipedia describes them as:

{==

In software engineering, a software design pattern is a general reusable solution 
to a commonly recurring problem within a given context in software design.
It is not a finished design that can be directly ported into source or machine code.
Rather, it is a description or template for how to solve a problem that can be used in many different situations.

==}

# 🚨 Be Careful!!

Developers, mostly beginners, makes the mistake of over-thinking and forcing the design patterns
which results in un-maintainable mess. The rule should always be to make the codebase as simple as possible.
Once you start developing you will start to see recurring patterns in your codebase, 
at which point you can start factoring your code using relevant design patterns.

- Design Patterns are not silver bullets to your problems. Use them consciously.
- Don't try to force them. Bad things might happen if done so!!
- Remember! Design Patterns are gudelines towards finding solutions to problem; not solutions to problems themselves.
  
## Types of Design Patterns

Adapted from the Gang-of-four (GoF) book on `Design Patterns`; 
there are broadly three types of useful & popular design patterns:

1. Creational
2. Structural 
3. Behavioral

## **Creational** Design Patterns

> In simple words, Creational design patterns are focused towards 
> how to instantiate an object or group of objects

Wikipedia says:

{==

In software engineering, Creational design patterns are design patterns that deal with **object creation mechanism**;
trying to create objects in a manners suitable for the sitation.
The basic form of object creation could result in design problems or added complexity to the design.
Creational design patetrns solve this problem by controlling this object creation.

==}

There are 6 types of CReational patterns:

1. Simple Factory
2. Factory Method
3. Abstract Factory
4. Builder
5. Prototpe
6. Singleton

### 🏠 Simple Factory

A real world example:
> Consider, you are building house and you need doors.
> You can either put on some carpenter clothes, bring some glue, wood, tools and make the door yourself.
> Or, you can call the factory and get the door delivered to you;
> so that you don't need to learn anything about door making or deal with the mess it brings.

In simple words,

> **Simple Factory** generates an instance for client without exposing any instantiation logic to the clients.

Wikipedia says,

{==

In Object oriented Programming language (OOP), **Factory** is an object for creating other objects.
Formally, a factory is a _function_ or _method_ that returns objects of a varying protoype 
or class from some method call, which is assumed to be `new`.

==}

** A Programatic example:**

First of all, we have a door interface and implementation of a wooden door.

```php
// The Simple Factory INTERFACE
// 
interface Door{
    public function get_width(): float;
    public function get_height(): float;
}

// A CONCRETE implementation of Door interface
//
class WoodenDoor implements Door{
    protected $width;
    protected $height;

    public function __construct(float $width, float $height){
        $this->$width = $width;
        $this->$height = $height;
    }

    public function get_width(): float{
        return $this->$width;
    }

    public function get_height(): float{
        return $this->$height;
    }
}
```

Then we have our door factory that makes the door and returns it.

```php
class DoorFactory{
    public static function make_door($width, $height): Door {
        return new WoodenDoor($height, $width);
    }
}
```

And then it can be used as:

```php
<?php

// Make a door of 100x200
$door = DoorFactory::make_door(100, 200);

echo "Width = " . $door->get_width();
echo "Height = " . $door->get_width();

// Make a door of 50x100
$door2 = DoorFactory::make_door(50, 100);

```







### 🏭 Factory Method

### 🛠 Abstract Factory

### 👷🏽 Builder

### 🐏 Prototype

### 💍 Singleton





