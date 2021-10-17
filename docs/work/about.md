<figure>
    <img src="https://4.bp.blogspot.com/-tgA9nKQJJ2Q/Vtra_9UwzDI/AAAAAAAANhQ/VmWebFhwBxw/s1600/underconstruction.jpg
" width="800" />
</figure>

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




<style>

* {
   box-sizing: border-box;
}

:root {
   --background: white;

   --primary: #ff1ead;
   --secondary: #1effc3;

   --card-size: 300px;
}

.card {
   width: calc(var(--card-size) * 1.586);
   height: var(--card-size);

   border-radius: 0.75rem;

   background: black;

   display: grid;
   grid-template-columns: 40% auto;
   color: white;

   align-items: center;

   will-change: transform;
   transition: transform 0.25s cubic-bezier(0.4, 0.0, 0.2, 1), box-shadow 0.25s cubic-bezier(0.4, 0.0, 0.2, 1);

   &:hover {
      transform: scale(1.1);
      box-shadow:  0 32px 80px 14px rgba(0,0,0,0.36), 0 0 0 1px rgba(0, 0, 0, 0.3);
   }
}

.card-details {
   padding: 1rem;
}

.name {
   font-size: 1.25rem;
}

.occupation {
   font-weight: 600;
   color: var(--primary);
}

.card-avatar {
   display: grid;
   place-items: center;
}

svg {
   fill: white;
   width: 65%;
}

.card-about {
   margin-top: 1rem;
   display: grid;
   grid-auto-flow: column;
}

.item {
   display: flex;
   flex-direction: column;
   margin-bottom: 0.5rem;

   .value {
      font-size: 1rem;
   }

   .label {
      margin-top: 0.15rem;
      font-size: 0.75rem;
      font-weight: 600;
      color: var(--primary);
   }
}

.skills {
   display: flex;
   flex-direction: column;
   margin-top: 0.75rem;

   .label {
      font-size: 1rem;
      font-weight: 600;
      color: var(--primary);
   }

   .value {
      margin-top: 0.15rem;
      font-size: 0.75rem;
      line-height: 1.25rem;
   }
}

</style>

<div class="card">
   <div class="card-avatar">
   </div>
   <div class="card-details">
      <div class="name">Saitama</div>
      <div class="occupation">Hero</div>
      
      <div class="card-about">
         <div class="item">
            <span class="value">25</span>
            <span class="label">Age</span>
         </div>
         <div class="item">
            <span class="value">70 kg </span>
            <span class="label">Weight</span>
         </div>
         <div class="item">
            <span class="value">175 cm</span>
            <span class="label">Height</span>
         </div>
      </div>
      <div class="skills">
         <span class="value">Immeasurable Physical Prowess, Supernatural Reflexes and Senses, Invulnerability, Indomitable Will, Enhanced Fighting Skill</span>
      </div>
   </div>
</div>








