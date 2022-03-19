## 🏠 Simple Factory

### A real world example:
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

### 👨🏻‍💻 **A Programatic example:**

First of all, we have a door interface and implementation of a wooden door.

```php
<?php
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
<?php

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

** ❓ When to use?**

{==

When creating an object is not just a few assignments, but involves some logic; 
it makes sense to put it in a dedicated factory instead of repeating same code everywhere.

==}



