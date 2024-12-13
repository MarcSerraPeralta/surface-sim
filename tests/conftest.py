def pytest_addoption(parser):
    parser.addoption("--show-figures", action="store_true")


def pytest_generate_tests(metafunc):
    # This is called for every test. Only get/set command line arguments
    # if the argument is specified in the list of test "fixturenames".
    # The hyphens in the arguments are substutited by underscores.
    option_value = metafunc.config.option.show_figures
    if "show_figures" in metafunc.fixturenames and option_value is not None:
        metafunc.parametrize("show_figures", [bool(option_value)])
