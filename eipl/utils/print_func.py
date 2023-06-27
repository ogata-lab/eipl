#
# Copyright (c) 2023 Ogata Laboratory, Waseda University
#
# Released under the AGPL license.
# see https://www.gnu.org/licenses/agpl-3.0.txt
#

OK = "\033[92m"
WARN = "\033[93m"
NG = "\033[91m"
END_CODE = "\033[0m"


def print_info(msg):
    print(OK + "[INFO] " + END_CODE + msg)


def print_warn(msg):
    print(WARN + "[WARNING] " + END_CODE + msg)


def print_error(msg):
    print(NG + "[ERROR] " + END_CODE + msg)
