from .version import __version__

def get_version():
    """Returns the version of aimped library."""
    return f'aimped version: {__version__}'


if __name__ == '__main__':
    print(get_aimped_version())