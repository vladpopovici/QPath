# -*- coding: utf-8 -*-

#############################################################################
# Copyright Vlad Popovici <popovici@bioxlab.org>
#
# Licensed under the MIT License. See LICENSE file in root folder.
#############################################################################

__author__ = "Vlad Popovici <popovici@bioxlab.org>"
__version__ = 0.1


class Error(Exception):
    """Basic error exception for QPATH.

    Args:
        msg (str): Human-readable string describing the exception.
        code (:obj:`int`, optional): Error code.

    Attributes:
        msg (str): Human-readable string describing the exception.
        code (int): Error code.
    """

    def __init__(self, msg, code=1, *args):
        self.message = "QPATH: " + msg
        self.code = code
        super(Error, self).__init__(msg, code, *args)
