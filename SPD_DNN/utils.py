import torch
import numpy as np


def symmetric(A):
    return 0.5 * (A + A.transpose(-2, -1))


# def orthogonal_projection(A, B):
#     out = A - B @ A.transpose(-2, -1) @ B
#     return out


def orthogonal_projection(A, B):
    out = A - B @ (symmetric(B.transpose(-2, -1) @ A))
    return out


#


def retraction(A, ref=None):

    if ref == None:
        data = A

    else:
        data = A + ref

    if A.shape[-2] < A.shape[-1]:
        data = data.transpose(-2, -1)

    Q, R = data.qr()

    sign = (R.diagonal(dim1=-2, dim2=-1).sign() + 0.5).sign().diag_embed()
    out = Q @ sign

    if A.shape[-2] < A.shape[-1]:
        out = out.transpose(-2, -1)


    return out
