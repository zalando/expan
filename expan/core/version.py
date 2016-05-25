#
__version__ = "0.2.4"


def version_numbers():
	return [int(x) for x in __version__.split('.')]


def git_commit_count():
	"""
  Returns the output of `git rev-list --count HEAD` as an int

  http://programmers.stackexchange.com/a/151558
  """
	print
	'dd'
	commit_count = None

	import subprocess
	output = subprocess.check_output(
		["git", "rev-list", "--count", "HEAD"])
	commit_count = int(output)

	return commit_count


def git_latest_commit():
	""""
  Returns output of `git rev-parse HEAD`

  http://programmers.stackexchange.com/a/151558
  """
	print
	'ee'
	latest_commit = None
	import subprocess
	output = subprocess.check_output(
		["git", "rev-parse", "HEAD"]).strip()

	return output


def version(format_str='{short}'):
	major, minor, patch = version_numbers()

	format_dict = {
		'major': major,
		'minor': minor,
		'patch': patch,
		'short': 'v{:d}.{:d}.{:d}'.format(major, minor, patch),
	}

	format_str = format_str.replace(
		'{commits', '{commit_count').replace(
		'{num_commits', '{commit_count').replace(
		'{last_commit', '{latest_commit').replace(
		'{commit', '{latest_commit').replace(
		'{hash', '{latest_commit').replace(
		'{HEAD', '{latest_commit').replace(
		'{long}', 'v{major:d}.{minor:d}.{patch:d}.{commit_count:d}-r{latest_commit}')

	if ('{commit_count' in format_str) or ('{latest_commit' in format_str):
		format_dict['commit_count'] = git_commit_count()
		format_dict['latest_commit'] = git_latest_commit()

	return format_str.format(**format_dict)


if __name__ == '__main__':
	print
	version('Short Version String: "{short}"')
	print
	version('Full Version String: "{long}"')
