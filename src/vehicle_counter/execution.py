"""Cooperative cancellation shared by pipeline and sequential experiments."""


class AnalysisCancelled(Exception):
    pass


def check_cancel(should_cancel=None):
    if should_cancel is not None and should_cancel():
        raise AnalysisCancelled("Analisis dibatalkan pengguna; hasil parsial belum lengkap.")
