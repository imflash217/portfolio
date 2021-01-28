# Notes
???+ success ""
    Lorem ipsum dolor sit amet, consectetur adipiscing elit. Nulla et euismod
    nulla. Curabitur feugiat, tortor non consequat finibus, justo purus auctor
    massa, nec semper lorem quam in massa.

    ``` python
    def bubble_sort(items):
        for i in range(len(items)):
            for j in range(len(items) - 1 - i):
                if items[j] > items[j + 1]:
                    items[j], items[j + 1] = items[j + 1], items[j]
    ```

    Nunc eu odio eleifend, blandit leo a, volutpat sapien. Phasellus posuere in
    sem ut cursus. Nullam sit amet tincidunt ipsum, sit amet elementum turpis.
    Etiam ipsum quam, mattis in purus vitae, lacinia fermentum enim.

!!! question "Pied Piper"
    Lorem ipsum dolor sit amet, consectetur adipiscing elit. Nulla et euismod
    nulla. Curabitur feugiat, tortor non consequat finibus, justo purus auctor
    massa, nec semper lorem quam in massa.

!!! quote "Quote"
    Lorem ipsum dolor sit amet, consectetur adipiscing elit. Nulla et euismod
    nulla. Curabitur feugiat, tortor non consequat finibus, justo purus auctor
    massa, nec semper lorem quam in massa.

!!! danger "Danger"
    Lorem ipsum dolor sit amet, consectetur adipiscing elit. Nulla et euismod
    nulla. Curabitur feugiat, tortor non consequat finibus, justo purus auctor
    massa, nec semper lorem quam in massa.

!!! success "Success"
    Lorem ipsum dolor sit amet, consectetur adipiscing elit. Nulla et euismod
    nulla. Curabitur feugiat, tortor non consequat finibus, justo purus auctor
    massa, nec semper lorem quam in massa.

[Submit :fontawesome-solid-paper-plane:](#){: .md-button .md-button--primary }

=== "C"

    ``` c
    #include <stdio.h>

    int main(void) {
      printf("Hello world!\n");
      return 0;
    }
    ```

=== "C++"

    ``` c++
    #include <iostream>

    int main(void) {
      std::cout << "Hello world!" << std::endl;
      return 0;
    }
    ```
------------------------------------------------------------
<!-- ##################################################################### -->

```python
#####################################################################
## प्रोजेक्ट-शिक्षा
#####################################################################

def विभाग(भाज्य, भाजक):
    भागफल = 0
    भाग = 1
    शेषफल = 0

    print(f"-----------------------------------")
    print(f"भाज्य - (भाजक x भाग) = शेष [?] भाजक")
    print(f"-----------------------------------")

    if भाज्य < भाजक:
        # print
        raise ValueError(f"भाज्य < भाजक [ग़लत संख्याएँ दी गयीं. कृपया सही संख्या अंकित करें.]")

    while True:
        शेष = भाज्य - (भाजक * भाग)
        if शेष >= भाजक:
            print(f"{भाज्य} - ({भाजक} x {भाग}) = {शेष} > {भाजक}")
            भाग = भाग + 1
        else:
            print(f"{भाज्य} - ({भाजक} x {भाग}) = {शेष} < {भाजक} .समाप्त")
            भागफल = भाग
            शेषफल = शेष
            print(f"-----------------------------------")
            return {"भागफल": भागफल, "शेषफल": शेषफल}

#####################################################################
```

![Placeholder](https://dummyimage.com/60x40/eee/aaa){: align=left }
lorem ipsum

----------

![Placeholder](https://dummyimage.com/100x100/eee/aaa){: loading=lazy }

----------

<figure>
  <img src="https://dummyimage.com/100x100/eee/aaa" width="100" />
  <figcaption>Image caption</figcaption>
</figure>

-------------
- [x] Lorem ipsum dolor sit amet, consectetur adipiscing elit
- [ ] Vestibulum convallis sit amet nisi a tincidunt
    * [x] In hac habitasse platea dictumst
    * [x] In scelerisque nibh non dolor mollis congue sed et metus
    * [ ] Praesent sed risus massa
- [ ] Aenean pretium efficitur erat, donec pharetra, ligula non scelerisque

-------------

$$
\operatorname{ker} f=\{g\in G:f(g)=e_{H}\}{\mbox{.}}
$$

-------------

The homomorphism $f$ is injective if and only if its kernel is only the
singleton set $e_G$, because otherwise $\exists a,b\in G$ with $a\neq b$ such
that $f(a)=f(b)$.

-------------

The HTML specification is maintained by the W3C.

*[HTML]: Hyper Text Markup Language
*[W3C]: World Wide Web Consortium

<!-- ##################################################################### -->

## Welcome to MkDocs

For full documentation visit [mkdocs.org](https://www.mkdocs.org).

Lorem ipsum[^1] dolor sit amet, consectetur adipiscing elit.[^2]
[^1]: Lorem ipsum dolor sit amet, consectetur adipiscing elit.
[^2]:
    Lorem ipsum dolor sit amet, consectetur adipiscing elit. Nulla et euismod

```python
#####################################################################
## प्रोजेक्ट-शिक्षा
#####################################################################

def विभाग(भाज्य, भाजक):
    भागफल = 0
    भाग = 1
    शेषफल = 0

    print(f"-----------------------------------")
    print(f"भाज्य - (भाजक x भाग) = शेष [?] भाजक")
    print(f"-----------------------------------")

    if भाज्य < भाजक:
        # print
        raise ValueError(f"भाज्य < भाजक [ग़लत संख्याएँ दी गयीं. कृपया सही संख्या अंकित करें.]")

    while True:
        शेष = भाज्य - (भाजक * भाग)
        if शेष >= भाजक:
            print(f"{भाज्य} - ({भाजक} x {भाग}) = {शेष} > {भाजक}")
            भाग = भाग + 1
        else:
            print(f"{भाज्य} - ({भाजक} x {भाग}) = {शेष} < {भाजक} .समाप्त")
            भागफल = भाग
            शेषफल = शेष
            print(f"-----------------------------------")
            return {"भागफल": भागफल, "शेषफल": शेषफल}

#####################################################################
```

![Placeholder](https://dummyimage.com/60x40/eee/aaa){: align=left }
lorem ipsum

----------

![Placeholder](https://dummyimage.com/100x100/eee/aaa){: loading=lazy }

----------

<figure>
  <img src="https://dummyimage.com/100x100/eee/aaa" width="100" />
  <figcaption>Image caption</figcaption>
</figure>

-------------
- [x] Lorem ipsum dolor sit amet, consectetur adipiscing elit
- [ ] Vestibulum convallis sit amet nisi a tincidunt
    * [x] In hac habitasse platea dictumst
    * [x] In scelerisque nibh non dolor mollis congue sed et metus
    * [ ] Praesent sed risus massa
- [ ] Aenean pretium efficitur erat, donec pharetra, ligula non scelerisque

-------------

$$
\operatorname{ker} f=\{g\in G:f(g)=e_{H}\}{\mbox{.}}
$$

-------------

The homomorphism $f$ is injective if and only if its kernel is only the
singleton set $e_G$, because otherwise $\exists a,b\in G$ with $a\neq b$ such
that $f(a)=f(b)$.

-------------

The HTML specification is maintained by the W3C.

*[HTML]: Hyper Text Markup Language
*[W3C]: World Wide Web Consortium
    nulla. Curabitur feugiat, tortor non consequat finibus, justo purus auctor
    massa, nec semper lorem quam in massa.

## Commands

* `mkdocs new [dir-name]` - Create a new project.
* `mkdocs serve` - Start the live-reloading docs server.
* `mkdocs build` - Build the documentation site.
* `mkdocs -h` - Print help message and exit.

## Project layout

    mkdocs.yml    # The configuration file.
    docs/
        index.md  # The documentation homepage.
        ...       # Other markdown pages, images and other files.

```python
import torch
import pytorch_lightning as pl
```

## Deployment
```
git add . && git commit -m "update" && git push -u origin main && mkdocs gh-deploy --force
```
<!--  -->