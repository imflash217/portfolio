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
$argadd docs/python/cookbook_dabeaz/ch08.md
set stal=2
tabnew
tabnew
tabnew
tabnew
tabnew
tabnew
tabnew
tabrewind
edit mkdocs.yml
argglobal
balt docs/blogs/lightning/about.md
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
let s:l = 71 - ((38 * winheight(0) + 20) / 40)
if s:l < 1 | let s:l = 1 | endif
keepjumps exe s:l
normal! zt
keepjumps 71
normal! 0
tabnext
edit docs/index.md
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
let s:l = 65 - ((39 * winheight(0) + 20) / 40)
if s:l < 1 | let s:l = 1 | endif
keepjumps exe s:l
normal! zt
keepjumps 65
normal! 06|
tabnext
edit docs/notes/about.md
argglobal
balt docs/index.md
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
let s:l = 1 - ((0 * winheight(0) + 19) / 38)
if s:l < 1 | let s:l = 1 | endif
keepjumps exe s:l
normal! zt
keepjumps 1
normal! 0
tabnext
edit docs/blogs/deep_learning/dl_04_09.md
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
let s:l = 1 - ((0 * winheight(0) + 19) / 38)
if s:l < 1 | let s:l = 1 | endif
keepjumps exe s:l
normal! zt
keepjumps 1
normal! 0
tabnext
edit docs/algorithms/dict.md
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
let s:l = 52 - ((36 * winheight(0) + 19) / 38)
if s:l < 1 | let s:l = 1 | endif
keepjumps exe s:l
normal! zt
keepjumps 52
normal! 03|
tabnext
edit docs/index.md
argglobal
balt docs/algorithms/dict.md
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
let s:l = 40 - ((21 * winheight(0) + 19) / 38)
if s:l < 1 | let s:l = 1 | endif
keepjumps exe s:l
normal! zt
keepjumps 40
normal! 033|
tabnext
edit docs/nlp/naive_bayes.md
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
let s:l = 15 - ((14 * winheight(0) + 19) / 38)
if s:l < 1 | let s:l = 1 | endif
keepjumps exe s:l
normal! zt
keepjumps 15
normal! 049|
tabnext
edit docs/algorithms/linked_list.md
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
let s:l = 12 - ((11 * winheight(0) + 19) / 38)
if s:l < 1 | let s:l = 1 | endif
keepjumps exe s:l
normal! zt
keepjumps 12
normal! 028|
tabnext 1
set stal=1
badd +1 mkdocs.yml
badd +1 docs/python/cookbook_dabeaz/ch08.md
badd +0 docs/index.md
badd +32 docs/notes/about.md
badd +1 docs/blogs/deep_learning/dl_04_09.md
badd +1 docs/algorithms/dict.md
badd +1 docs/nlp/naive_bayes.md
badd +45 docs/algorithms/linked_list.md
badd +1 docs/blogs/lightning/about.md
badd +1 docs/awesome.md
badd +23 docs/nlp/regex.md
badd +17 docs/nlp/nlp_book.md
badd +851 docs/blogs/deep_learning/einops2.md
badd +721 docs/algorithms/binary_tree.md
badd +32 docs/stylesheets/extra.css
badd +8 docs/transformers/about.md
badd +1 docs/algorithms/QA.md
badd +1 docs/blogs/about.md
badd +24 docs/todo.md
badd +1 docs/blogs/lightning/tut_1.md
badd +111 ~/.vimrc
badd +1 docs/notes/ECE542/ece542_hw1a.md
badd +1 docs/gists/about.md
badd +2 docs/gists/lightning/api/configure_optimizers.md
badd +6 docs/gists/lightning/api/forward.md
badd +1 docs/gists/python_snippets.md
badd +105 docs/blogs/physics/blog_01282021.md
badd +10 docs/algorithms/023_add_lists.md
badd +9 docs/algorithms/024_depth_first_values.md
badd +9 docs/algorithms/025_breadth_first_values.md
badd +10 docs/algorithms/026_tree_includes.md
badd +10 docs/algorithms/027_tree_sum.md
badd +8 docs/algorithms/028_tree_min_value.md
badd +7 docs/blogs/lightning/api.md
badd +6 docs/gists/lightning/api/freeze.md
badd +1 docs/gists/lightning/api/log.md
badd +1 docs/gists/lightning/api/training_step.md
badd +70 docs/nlp/CS224N/cs224n_1.md
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
