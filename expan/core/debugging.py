'''
This started as a very simple central place to provide a really easy interface
to output (standardised across projects) so I could always just do dbg(1, 'blah').

It has morphed and mutated into an ugly wrapper on top of logger.

I keep it because I still find logger to be a PITA and too complicated to use
quickly.

A few things it offers over just using logging:
	- defaults to DEBUG level
	- use positive and negative numbers to configure level
	- standardised logging msg
	- easier calling `dbg(1, 'blah')` - dubious advantage

Created on Aug 11, 2014

TODO: need to figure out how to tell the logging class to skip this module's functions
in the stack, for printing of module and function name etc.. Could potentially
fiddle with the stack before calling logging, but that's very dodgy.

@author: rmuil
'''
import inspect
import logging
import os

default_dbg_lvl = 2

_stackframe_filename_idx = 1
_stackframe_lineno_idx = 2
_stackframe_funcname_idx = 3


def _wherefrom(skip_stack_lvls=0):
	"""
	Returns the name of the function from which this was called.

	NB: from https://docs.python.org/2/library/inspect.html:
	"Keeping references to frame objects, as found in the first element of the
	frame records these functions return, can cause your program to create
	reference cycles..."

	In the stack, pos 0 is this function, 1 is direct caller of this function
	and 2 is the caller of the caller, which would usually be the interesting one.

	NB: this seems to throw an IndexError at the inspect.stack() line sometimes. HAven't traced the bug yet.
	"""
	stack_lvl = 1 + skip_stack_lvls
	caller_name = caller_file = None
	caller_lineno = -1
	current_stack = inspect.stack()
	try:
		caller_name = current_stack[stack_lvl][_stackframe_funcname_idx]
		caller_file = os.path.basename(current_stack[stack_lvl][_stackframe_filename_idx])
		caller_lineno = current_stack[stack_lvl][_stackframe_lineno_idx]
	finally:
		del current_stack

	if caller_file.startswith('<ipython-input'):
		caller_file = '<ipython>'
		caller_name = '<prompt>'

	return (caller_name, caller_file, caller_lineno)


class Dbg():
	def __init__(self, logger=None, dbg_lvl=default_dbg_lvl):
		if logger is None:
			logging.basicConfig(level=logging.DEBUG)  # configure the root logger
			logger = logging.getLogger()  # get the root logger
		self.logger = logger
		self._dbg_lvl = dbg_lvl
		self._call_count = {}

	def __call__(self, lvl, msg, skip_stack_lvls=0):
		return self.out(lvl, msg, skip_stack_lvls=skip_stack_lvls + 1)

	def out(self, lvl, msg,
			skip_stack_lvls=0):
		"""
		This isn't very well implemented for important messages...
		we should move to relying much more on the Logging class for dispatching
		and filtering logs.

		Treat this as just a convenience.
		"""
		caller_name, caller_file, caller_lineno = _wherefrom(skip_stack_lvls + 1)
		dlvl = self._dbg_lvl
		if caller_file not in self._call_count:
			self._call_count[caller_file] = 1
		else:
			self._call_count[caller_file] += 1
		if lvl <= dlvl:
			if lvl == 0:
				self.logger.info('%s,%d|%s: %s' % (caller_file, caller_lineno, caller_name, msg))
			else:
				if lvl == -1:
					self.logger.warning('%s,%d|%s: %s' % (caller_file, caller_lineno, caller_name, msg))
				elif lvl < -1:
					self.logger.error('%s,%d|%s: %s' % (caller_file, caller_lineno, caller_name, msg))
				else:
					self.logger.debug('D%d|%s,%d|%s: %s' % (lvl, caller_file, caller_lineno, caller_name, msg))

	def set_lvl(self, new_dbg_lvl):
		self._dbg_lvl = new_dbg_lvl

	def get_lvl(self):
		return self._dbg_lvl

	def reset(self):
		self._dbg_lvl = default_dbg_lvl
		self._call_count = {}
