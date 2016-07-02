package = 'nninit'
version = '0.0-0'

source = {
	-- Use this for quick installation from local sources during development.
	url = '.',
	dir = '.',
	-- url = 'git://github.com/adityaramesh/lantern',
	-- branch = 'master'
}

description = {
	summary = "A collection of initialization schemes for neural networks.",
	homepage = 'https://github.com/adityaramesh/nninit',
	license  = 'BSD 3-Clause'
}

dependencies = {
	'torch >= 7.0',
	'nn',
	'image',
	'f-strings'
}

build = {
	type = 'builtin',
	modules = {['nninit'] = 'source/nninit.lua'}
}
