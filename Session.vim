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
tabnew
tabnew
tabnew
tabnew
tabnew
tabrewind
edit mkdocs.yml
argglobal
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
let s:l = 116 - ((15 * winheight(0) + 18) / 36)
if s:l < 1 | let s:l = 1 | endif
keepjumps exe s:l
normal! zt
keepjumps 116
normal! 040|
tabnext
edit docs/blogs/prob/intro.md
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
let s:l = 51 - ((18 * winheight(0) + 18) / 36)
if s:l < 1 | let s:l = 1 | endif
keepjumps exe s:l
normal! zt
keepjumps 51
normal! 056|
tabnext
edit docs/stylesheets/extra.css
argglobal
balt docs/blogs/prob/intro.md
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
let s:l = 31 - ((17 * winheight(0) + 18) / 36)
if s:l < 1 | let s:l = 1 | endif
keepjumps exe s:l
normal! zt
keepjumps 31
normal! 081|
tabnext
edit docs/blogs/deep_learning/blog_tf_v1.md
argglobal
balt docs/blogs/prob/intro.md
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
let s:l = 1 - ((0 * winheight(0) + 18) / 36)
if s:l < 1 | let s:l = 1 | endif
keepjumps exe s:l
normal! zt
keepjumps 1
normal! 0
tabnext
edit docs/blogs/deep_learning/blog_einops.md
argglobal
balt docs/blogs/deep_learning/blog_tf_v1.md
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
let s:l = 9 - ((8 * winheight(0) + 18) / 36)
if s:l < 1 | let s:l = 1 | endif
keepjumps exe s:l
normal! zt
keepjumps 9
normal! 0
tabnext
edit docs/index.md
argglobal
balt docs/blogs/deep_learning/blog_einops.md
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
let s:l = 33 - ((24 * winheight(0) + 18) / 36)
if s:l < 1 | let s:l = 1 | endif
keepjumps exe s:l
normal! zt
keepjumps 33
normal! 017|
tabnext
edit docs/python/design_patterns.md
argglobal
balt docs/blogs/deep_learning/blog_einops.md
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
let s:l = 291 - ((34 * winheight(0) + 18) / 36)
if s:l < 1 | let s:l = 1 | endif
keepjumps exe s:l
normal! zt
keepjumps 291
normal! 0
tabnext
edit docs/python/decorators.md
argglobal
balt docs/python/design_patterns.md
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
let s:l = 34 - ((13 * winheight(0) + 18) / 36)
if s:l < 1 | let s:l = 1 | endif
keepjumps exe s:l
normal! zt
keepjumps 34
normal! 028|
tabnext 2
badd +1 mkdocs.yml
badd +0 docs/blogs/prob/intro.md
badd +1 docs/blogs/deep_learning/blog_tf_v1.md
badd +1 docs/blogs/deep_learning/blog_einops.md
badd +1 docs/index.md
badd +1 docs/python/design_patterns.md
badd +1 docs/python/decorators.md
badd +29 ~/.vimrc
badd +1 docs/publications/about.md
badd +32 docs/notes/about.md
badd +1 docs/blogs/physics/blog_01282021.md
badd +1 ~/Desktop/flashAI/fluent_python/src/notebook_css/style-notebook.css
badd +29 docs/stylesheets/extra.css
badd +1 docs/
badd +311 docs/python/cookbook_dabeaz/ch01.md
badd +92 docs/blogs/algorithms/blog_01292021_hashtables.md
badd +1 docs/blogs/algorithms/blog_01312021_sorting.md
badd +26 docs/blogs/algorithms/blog_02012021_document_distance.md
badd +17 docs/blogs/algorithms/blog_02012021_priority_queue.md
badd +78 docs/blogs/algorithms/blog_03152021_arrays.md
badd +1 docs/algorithms/023_add_lists.md
badd +68 docs/algorithms/024_depth_first_values.md
badd +80 docs/algorithms/025_breadth_first_values.md
badd +14 docs/algorithms/026_tree_includes.md
badd +9 docs/algorithms/027_tree_sum.md
badd +1 docs/algorithms/028_tree_min_value.md
badd +18 docs/gists/lightning/api/configure_optimizers.md
badd +101 ~/.vim_runtime/my_configs.vim
badd +99 ~/.vim_runtime/vimrcs/plugins_config.vim
badd +1 ~/.vim_runtime/vimrcs/basic.vim
badd +183 ~/.vim_runtime/vimrcs/extended.vim
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
let g:this_session = v:this_session
let g:this_obsession = v:this_session
doautoall SessionLoadPost
unlet SessionLoad
" vim: set ft=vim :
