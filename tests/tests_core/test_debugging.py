import logging
import logging.config
import os

import expan.core.debugging as dbgcls

reload(dbgcls)


def test_default_debugging(capsys):
	"""
	Test the debugging object created with default options.
	"""
	dbg = dbgcls.Dbg()
	dbg(-2, 'This should be an error')
	out, err = capsys.readouterr()
	print
	'out is: ' + out
	print
	'err is: ' + err
	errparts = err.split('|')
	print
	'errparts[1] is: [{}]'.format(errparts[1])
	assert (errparts[1] == 'test_default_debugging: This should be an error\n')

	dbg(-1, 'This should be a warning')
	dbg(0, 'This should be a stdout')
	# TODO: capsys appends output to existing, so we can't test each call's output individually.
	# out,err = capsys.readouterr()
	# assert(err.split('|')[1] == 'test_default_debugging: This should be a stdout')
	dbg(1, 'This should be a debug message')
	dbg(2, 'This should be a level 2 dbg message!')

	dbg.out(1, 'can also use like this!')


def test_debugging_with_file_config():
	"""
	Test the debugging object created with options from logging config file.
	TODO: test the output!
	"""
	logging.config.fileConfig(os.getcwd() + '/tests/tests_core/test-logging.conf')
	logger = logging.getLogger('simple')
	dbgf = dbgcls.Dbg(logger)

	dbgf(-2, 'This should be an error')
	dbgf(-1, 'This should be a warning')
	dbgf(0, 'This should be a stdout')
	dbgf(1, 'This should be a debug message')
	dbgf(2, 'This should be a level 2 dbg message!')

	dbgf.out(1, 'can also use like this!')


if __name__ == "__main__":
	pass  # TODO: call py.test?
