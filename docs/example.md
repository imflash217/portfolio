```
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
