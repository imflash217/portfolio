let SessionLoad = 1
if &cp | set nocp | endif
let s:so_save = &g:so | let s:siso_save = &g:siso | setg so=0 siso=0 | setl so=-1 siso=-1
let v:this_session=expand("<sfile>:p")
silent only
silent tabonly
cd ~/Desktop/flashAI/portfolio
if expand('%') == '' && !&modified && line('$') <= 1 && getline(1) == ''
  let s:wipebuf = bufnr('%')
endif
set shortmess=aoO
argglobal
%argdel
$argadd mkdocs.yml
tabnew
tabnew
tabrewind
edit docs/python/cookbook_dabeaz/ch07.md
argglobal
balt mkdocs.yml
setlocal fdm=manual
setlocal fde=0
setlocal fmr={{{,}}}
setlocal fdi=#
setlocal fdl=0
setlocal fml=1
setlocal fdn=20
setlocal fen
silent! normal! zE
let &fdl = &fdl
let s:l = 587 - ((35 * winheight(0) + 18) / 36)
if s:l < 1 | let s:l = 1 | endif
keepjumps exe s:l
normal! zt
keepjumps 587
normal! 0
tabnext
edit docs/python/cookbook_dabeaz/ch08.md
argglobal
balt docs/python/cookbook_dabeaz/ch07.md
setlocal fdm=manual
setlocal fde=0
setlocal fmr={{{,}}}
setlocal fdi=#
setlocal fdl=0
setlocal fml=1
setlocal fdn=20
setlocal fen
silent! normal! zE
let &fdl = &fdl
let s:l = 86 - ((8 * winheight(0) + 18) / 36)
if s:l < 1 | let s:l = 1 | endif
keepjumps exe s:l
normal! zt
keepjumps 86
normal! 030|
tabnext
edit mkdocs.yml
argglobal
balt docs/python/cookbook_dabeaz/ch07.md
setlocal fdm=manual
setlocal fde=0
setlocal fmr={{{,}}}
setlocal fdi=#
setlocal fdl=0
setlocal fml=1
setlocal fdn=20
setlocal fen
silent! normal! zE
let &fdl = &fdl
let s:l = 91 - ((7 * winheight(0) + 18) / 36)
if s:l < 1 | let s:l = 1 | endif
keepjumps exe s:l
normal! zt
keepjumps 91
normal! 02|
tabnext 2
badd +129 mkdocs.yml
badd +1 docs/python/cookbook_dabeaz/ch07.md
badd +32 docs/notes/about.md
badd +9 docs/projects/about.md
badd +20 docs/work/about.md
badd +40 ~/.vim_runtime/my_configs.vim
badd +11 ~/.vimrc
badd +1 ../../stylesheets/extra.css
badd +50 docs/index.md
badd +3 docs/paper_reviews/about.md
badd +0 docs/paper_reviews/dter.md
badd +5 docs/paper_reviews/detr.md
badd +0 docs/python/cookbook_dabeaz/ch08.md
if exists('s:wipebuf') && len(win_findbuf(s:wipebuf)) == 0
  silent exe 'bwipe ' . s:wipebuf
endif
unlet! s:wipebuf
set winheight=1 winwidth=20 shortmess=filnxtToOS
let s:sx = expand("<sfile>:p:r")."x.vim"
if filereadable(s:sx)
  exe "source " . fnameescape(s:sx)
endif
let &g:so = s:so_save | let &g:siso = s:siso_save
nohlsearch
let g:this_session = v:this_session
let g:this_obsession = v:this_session
doautoall SessionLoadPost
unlet SessionLoad
" vim: set ft=vim :
