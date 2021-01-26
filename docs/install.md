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