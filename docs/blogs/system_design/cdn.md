<!-- ---
hide:
  - navigation # Hide navigation
  - toc        # Hide table of contents
--- -->

# **CDN** अर्थात् *Content Delivery Network*

**CDN** अर्थात् servers का एक ऐसा network जो पृथ्वी पर अलग-अलग भूगौलिक क्षेत्रों में होता है; और इनका एक ही कार्य होता है: "static content जैसे images, videos, CSS, JavaScript, etc. को माँगने वाले client (जैसे apps, end-users, etc) तक पहुँचाना"। इस कार्य को करने के लिए CDN के द्वारा प्रयुक्त होने वाले technology का नाम है: **Dynamic Content Caching** जो `request path`, `query strings`, `cookies` और `request headers` के आधार पर HTML pages को **cache** करने की योग्यता प्रदान करता है।

> सरल भाषा में **CDN** मुख्यतः दो कार्यों को करता है:
> 
> - Client को *static content* देना 
> - नज़दीकी client की सर्वप्रथम सेवा करना (अर्थात्, अगर हमारा CDN `चेन्नई` में है और दो client, एक `बैंगलुरु` व दूसरा `दिल्ली` से CDN से एक website माँग रहे हैं तो सर्वप्रथम `बैंगलुरु` के `request` को पूरा किया जाएगा फिर `दिल्ली` को क्योंकि `दिल्ली` की अपेक्षा `बैंगलुरु` हमारे CDN के नज़दीक है जो `चेन्नई` में है)।

 