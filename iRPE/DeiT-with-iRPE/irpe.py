"""The implementation of iRPE (image relative position encoding)."""
from easydict import EasyDict as edict
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
try:
    from rpe_ops.rpe_index import RPEIndexFunction
except ImportError:
    RPEIndexFunction = None
    import warnings
    RED_STR = "\033[91m{}\033[00m"
    warnings.warn(RED_STR.format("[WARNING] The module `rpe_ops` is not built. \
For better training performance, please build `rpe_ops`."),)


@torch.no_grad()
def piecewise_index(relative_position, alpha, beta, gamma, dtype):
    """piecewise index function defined in Eq. (18) in our paper.

    Parameters
    ----------
    relative_position: torch.Tensor, dtype: long or float
        The shape of `relative_position` is (L, L).
    alpha, beta, gamma: float
        The coefficients of piecewise index function.

    Returns
    -------
    idx: torch.Tensor, dtype: long
        A tensor indexing relative distances to corresponding encodings.
        `idx` is a long tensor, whose shape is (L, L) and each element is in [-beta, beta].
    """
    rp_abs = relative_position.abs()
    mask = rp_abs <= alpha
    not_mask = ~mask
    rp_out = relative_position[not_mask]
    rp_abs_out = rp_abs[not_mask]
    y_out = (torch.sign(rp_out) * (alpha +
                                   torch.log(rp_abs_out / alpha) /
                                   math.log(gamma / alpha) *
                                   (beta - alpha)).round().clip(max=beta)).to(dtype)

    idx = relative_position.clone()
    if idx.dtype in [torch.float32, torch.float64]:
        # round(x) when |x| <= alpha
        idx = idx.round().to(dtype)

    # assign the value when |x| > alpha
    idx[not_mask] = y_out
    return idx


def get_absolute_positions(height, width, dtype, device):
    '''Get absolute positions

    Take height = 3, width = 3 as an example:
    rows:        cols:
    1 1 1        1 2 3
    2 2 2        1 2 3
    3 3 3        1 2 3

    return stack([rows, cols], 2)

    Parameters
    ----------
    height, width: int
        The height and width of feature map
    dtype: torch.dtype
        the data type of returned value
    device: torch.device
        the device of returned value

    Return
    ------
    2D absolute positions: torch.Tensor
        The shape is (height, width, 2),
        where 2 represents a 2D position (row, col).
    '''
    rows = torch.arange(height, dtype=dtype, device=device).view(
        height, 1).repeat(1, width)
    cols = torch.arange(width, dtype=dtype, device=device).view(
        1, width).repeat(height, 1)
    return torch.stack([rows, cols], 2)


@torch.no_grad()
def quantize_values(values):
    """Quantization: Map all values (long or float) into a discrte integer set.

    Parameters
    ----------
    values: torch.Tensor, dtype: long or float
        arbitrary shape

    Returns
    -------
    res: torch.Tensor, dtype: long
        The quantization result starts at 0.
        The shape is the same as that of `values`.
    uq.numel(): long
        The number of the quantization integers, namely `res` is in [0, uq.numel()).
    """
    # quantize and re-assign bucket id
    res = torch.empty_like(values)
    uq = values.unique()
    cnt = 0
    for (tid, v) in enumerate(uq):
        mask = (values == v)
        cnt += torch.count_nonzero(mask)
        res[mask] = tid
    assert cnt == values.numel()
    return res, uq.numel()


class METHOD:
    """define iRPE method IDs
    We divide the implementation of CROSS into CROSS_ROWS and CROSS_COLS.

    """
    EUCLIDEAN = 0
    QUANT = 1
    PRODUCT = 3
    CROSS = 4
    CROSS_ROWS = 41
    CROSS_COLS = 42


@torch.no_grad()
def _rp_2d_euclidean(diff, **kwargs):
    """2D RPE with Euclidean method.

    Parameters
    ----------
    diff: torch.Tensor
        The shape of `diff` is (L, L, 2),
        where L is the sequence length,
        and 2 represents a 2D offset (row_offset, col_offset).

    Returns
    -------
    index: torch.Tensor, dtype: long
        index to corresponding encodings.
        The shape of `index` is (L, L),
        where L is the sequence length.
    """
    dis = diff.square().sum(2).float().sqrt().round()
    return piecewise_index(dis, **kwargs)


@torch.no_grad()
def _rp_2d_quant(diff, **kwargs):
    """2D RPE with Quantization method.

    Parameters
    ----------
    diff: torch.Tensor
        The shape of `diff` is (L, L, 2),
        where L is the sequence length,
        and 2 represents a 2D offset (row_offset, col_offset).

    Returns
    -------
    index: torch.Tensor, dtype: long
        index to corresponding encodings.
        The shape of `index` is (L, L),
        where L is the sequence length.
    """

    dis = diff.square().sum(2)
    return piecewise_index(dis, **kwargs)


@torch.no_grad()
def _rp_2d_product(diff, **kwargs):
    """2D RPE with Product method.

    Parameters
    ----------
    diff: torch.Tensor
        The shape of `diff` is (L, L, 2),
        where L is the sequence length,
        and 2 represents a 2D offset (row_offset, col_offset).

    Returns
    -------
    index: torch.Tensor, dtype: long
        index to corresponding encodings.
        The shape of `index` is (L, L),
        where L is the sequence length.
    """
    # convert beta to an integer since beta is a float number.
    beta_int = int(kwargs['beta'])
    S = 2 * beta_int + 1
    # the output of piecewise index function is in [-beta_int, beta_int]
    r = piecewise_index(diff[:, :, 0], **kwargs) + \
        beta_int  # [0, 2 * beta_int]
    c = piecewise_index(diff[:, :, 1], **kwargs) + \
        beta_int  # [0, 2 * beta_int]
    pid = r * S + c
    return pid


@torch.no_grad()
def _rp_2d_cross_rows(diff, **kwargs):
    """2D RPE with Cross for rows.

    Parameters
    ----------
    diff: torch.Tensor
        The shape of `diff` is (L, L, 2),
        where L is the sequence length,
        and 2 represents a 2D offset (row_offset, col_offset).

    Returns
    -------
    index: torch.Tensor, dtype: long
        index to corresponding encodings.
        The shape of `index` is (L, L),
        where L is the sequence length.
    """
    dis = diff[:, :, 0]
    return piecewise_index(dis, **kwargs)


@torch.no_grad()
def _rp_2d_cross_cols(diff, **kwargs):
    """2D RPE with Cross for columns.

    Parameters
    ----------
    diff: torch.Tensor
        The shape of `diff` is (L, L, 2),
        where L is the sequence length,
        and 2 represents a 2D offset (row_offset, col_offset).

    Returns
    -------
    index: torch.Tensor, dtype: long
        index to corresponding encodings.
        The shape of `index` is (L, L),
        where L is the sequence length.
    """

    dis = diff[:, :, 1]
    return piecewise_index(dis, **kwargs)


# Define a mapping from METHOD_ID to Python function
_METHOD_FUNC = {
    METHOD.EUCLIDEAN: _rp_2d_euclidean,
    METHOD.QUANT: _rp_2d_quant,
    METHOD.PRODUCT: _rp_2d_product,
    METHOD.CROSS_ROWS: _rp_2d_cross_rows,
    METHOD.CROSS_COLS: _rp_2d_cross_cols,
}


def get_num_buckets(method, alpha, beta, gamma):
    """ Get number of buckets storing relative position encoding.
    The buckets does not contain `skip` token.

    Parameters
    ----------
    method: METHOD
        The method ID of image relative position encoding.
    alpha, beta, gamma: float
        The coefficients of piecewise index function.

    Returns
    -------
    num_buckets: int
        The number of buckets storing relative position encoding.
    """
    beta_int = int(beta)
    if method == METHOD.PRODUCT:
        # IDs in [0, (2 * beta_int + 1)^2) for Product method
        num_buckets = (2 * beta_int + 1) ** 2
    else:
        # IDs in [-beta_int, beta_int] except of Product method
        num_buckets = 2 * beta_int + 1
    return num_buckets


# (method, alpha, beta, gamma) -> (bucket_ids, num_buckets, height, width)
BUCKET_IDS_BUF = dict()


@torch.no_grad()
def get_bucket_ids_2d_without_skip(method, height, width,
                                   alpha, beta, gamma,
                                   dtype=torch.long, device=torch.device('cpu')):
    """Get bucket IDs for image relative position encodings without skip token

    Parameters
    ----------
    method: METHOD
        The method ID of image relative position encoding.
    height, width: int
        The height and width of the feature map.
        The sequence length is equal to `height * width`.
    alpha, beta, gamma: float
        The coefficients of piecewise index function.
    dtype: torch.dtype
        the data type of returned `bucket_ids`
    device: torch.device
        the device of returned `bucket_ids`

    Returns
    -------
    bucket_ids: torch.Tensor, dtype: long
        The bucket IDs which index to corresponding encodings.
        The shape of `bucket_ids` is (skip + L, skip + L),
        where `L = height * wdith`.
    num_buckets: int
        The number of buckets including `skip` token.
    L: int
        The sequence length
    """

    key = (method, alpha, beta, gamma, dtype, device)
    value = BUCKET_IDS_BUF.get(key, None)
    if value is None or value[-2] < height or value[-1] < width:
        if value is None:
            max_height, max_width = height, width
        else:
            max_height = max(value[-2], height)
            max_width = max(value[-1], width)
        # relative position encoding mapping function
        func = _METHOD_FUNC.get(method, None)
        if func is None:
            raise NotImplementedError(
                f"[Error] The method ID {method} does not exist.")
        pos = get_absolute_positions(max_height, max_width, dtype, device)

        # compute the offset of a pair of 2D relative positions
        max_L = max_height * max_width
        pos1 = pos.view((max_L, 1, 2))
        pos2 = pos.view((1, max_L, 2))
        # diff: shape of (L, L, 2)
        diff = pos1 - pos2

        # bucket_ids: shape of (L, L)
        bucket_ids = func(diff, alpha=alpha, beta=beta,
                          gamma=gamma, dtype=dtype)
        beta_int = int(beta)
        if method != METHOD.PRODUCT:
            bucket_ids += beta_int
        bucket_ids = bucket_ids.view(
            max_height, max_width, max_height, max_width)

        num_buckets = get_num_buckets(method, alpha, beta, gamma)
        value = (bucket_ids, num_buckets, height, width)
        BUCKET_IDS_BUF[key] = value
    L = height * width
    bucket_ids = value[0][:height, :width, :height, :width].reshape(L, L)
    num_buckets = value[1]

    return bucket_ids, num_buckets, L


@torch.no_grad()
def get_bucket_ids_2d(method, height, width,
                      skip, alpha, beta, gamma,
                      dtype=torch.long, device=torch.device('cpu')):
    """Get bucket IDs for image relative position encodings

    Parameters
    ----------
    method: METHOD
        The method ID of image relative position encoding.
    height, width: int
        The height and width of the feature map.
        The sequence length is equal to `height * width`.
    skip: int
        The number of skip token before spatial tokens.
        When skip is 0, no classification token.
        When skip is 1, there is a classification token before spatial tokens.
        When skip > 1, there are `skip` extra tokens before spatial tokens.
    alpha, beta, gamma: float
        The coefficients of piecewise index function.
    dtype: torch.dtype
        the data type of returned `bucket_ids`
    device: torch.device
        the device of returned `bucket_ids`

    Returns
    -------
    bucket_ids: torch.Tensor, dtype: long
        The bucket IDs which index to corresponding encodings.
        The shape of `bucket_ids` is (skip + L, skip + L),
        where `L = height * wdith`.
    num_buckets: int
        The number of buckets including `skip` token.
    """
    bucket_ids, num_buckets, L = get_bucket_ids_2d_without_skip(method, height, width,
                                                                alpha, beta, gamma,
                                                                dtype, device)

    # add an extra encoding (id = num_buckets) for the classification token
    if skip > 0:
        new_bids = bucket_ids.new_empty(size=(skip + L, skip + L))

        # if extra token exists, we add extra bucket as its encoding.
        extra_bucket_id = num_buckets
        num_buckets += 1

        new_bids[:skip] = extra_bucket_id
        new_bids[:, :skip] = extra_bucket_id
        new_bids[skip:, skip:] = bucket_ids

        bucket_ids = new_bids
    bucket_ids = bucket_ids.contiguous()
    return bucket_ids, num_buckets


class iRPE(nn.Module):
    """The implementation of image relative position encoding (excluding Cross method).

    Parameters
    ----------
    head_dim: int
        The dimension for each head.
    num_heads: int
        The number of parallel attention heads.
    mode: str or None
        The mode of image relative position encoding.
        Choices: [None, 'bias', 'contextual']
    method: METHOD
        The method ID of image relative position encoding.
        The `METHOD` class is defined in `irpe.py`.
    transposed: bool
        Whether to transpose the input feature.
        For iRPE on queries or keys, transposed should be `True`.
        For iRPE on values, transposed should be `False`.
    num_buckets: int
        The number of buckets, which store encodings.
    initializer: None or an inplace function
        [Optional] The initializer to `lookup_table`.
        Initalize `lookup_table` as zero by default.
    rpe_config: RPEConfig
        The config generated by the function `get_single_rpe_config`.
    """
    # a buffer to store bucket index
    # (key, rp_bucket, _ctx_rp_bucket_flatten)
    _rp_bucket_buf = (None, None, None)

    def __init__(self, head_dim, num_heads=8,
                 mode=None, method=None,
                 transposed=True, num_buckets=None,
                 initializer=None, rpe_config=None):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim

        # relative position
        assert mode in [None, 'bias', 'contextual']
        self.mode = mode

        assert method is not None, 'method should be a METHOD ID rather than None'
        self.method = method

        self.transposed = transposed
        self.num_buckets = num_buckets

        if initializer is None:
            def initializer(x): return None
        self.initializer = initializer

        self.reset_parameters()

        self.rpe_config = rpe_config

    @torch.no_grad()
    def reset_parameters(self):
        # initialize the parameters of iRPE
        if self.transposed:
            if self.mode == 'bias':
                self.lookup_table_bias = nn.Parameter(
                    torch.zeros(self.num_heads, self.num_buckets))
                self.initializer(self.lookup_table_bias)
            elif self.mode == 'contextual':
                self.lookup_table_weight = nn.Parameter(
                    torch.zeros(self.num_heads,
                                self.head_dim, self.num_buckets))
                self.initializer(self.lookup_table_weight)
        else:
            if self.mode == 'bias':
                raise NotImplementedError(
                    "[Error] Bias non-transposed RPE does not exist.")
            elif self.mode == 'contextual':
                self.lookup_table_weight = nn.Parameter(
                    torch.zeros(self.num_heads,
                                self.num_buckets, self.head_dim))
                self.initializer(self.lookup_table_weight)

    def forward(self, x, height=None, width=None):
        """forward function for iRPE.

        Parameters
        ----------
        x: torch.Tensor
            Input Tensor whose shape is (B, H, L, head_dim),
            where B is batch size,
                  H is the number of heads,
                  L is the sequence length,
                    equal to height * width (+1 if class token exists)
                  head_dim is the dimension of each head

        Returns
        -------
        rpe_encoding: torch.Tensor
            image Relative Position Encoding,
            whose shape is (B, H, L, L)
        """
        rp_bucket, self._ctx_rp_bucket_flatten = \
            self._get_rp_bucket(x, height=height, width=width)
        if self.transposed:
            return self.forward_rpe_transpose(x, rp_bucket)
        return self.forward_rpe_no_transpose(x, rp_bucket)

    def _get_rp_bucket(self, x, height=None, width=None):
        """Get relative position encoding buckets IDs corresponding the input shape

        Parameters
        ----------
        x: torch.Tensor
            Input Tensor whose shape is (B, H, L, head_dim),
            where B is batch size,
                  H is the number of heads,
                  L is the sequence length,
                    equal to height * width (+1 if class token exists)
                  head_dim is the dimension of each head
        height: int or None
            [Optional] The height of the input
            If not defined, height = floor(sqrt(L))
        width: int or None
            [Optional] The width of the input
            If not defined, width = floor(sqrt(L))

        Returns
        -------
        rp_bucket: torch.Tensor
            relative position encoding buckets IDs
            The shape is (L, L)
        _ctx_rp_bucket_flatten: torch.Tensor or None
            It is a private tensor for efficient computation.
        """
        B, H, L, D = x.shape
        device = x.device
        if height is None:
            E = int(math.sqrt(L))
            height = width = E
        key = (height, width, device)
        # use buffer if the spatial shape and device is not changable.

        if self._rp_bucket_buf[0] == key:
            return self._rp_bucket_buf[1:3]

        skip = L - height * width
        config = self.rpe_config
        if RPEIndexFunction is not None and self.mode == 'contextual' and self.transposed:
            # RPEIndexFunction uses int32 index.
            dtype = torch.int32
        else:
            dtype = torch.long
        rp_bucket, num_buckets = get_bucket_ids_2d(method=self.method,
                                                   height=height, width=width,
                                                   skip=skip, alpha=config.alpha,
                                                   beta=config.beta, gamma=config.gamma,
                                                   dtype=dtype, device=device)
        assert num_buckets == self.num_buckets

        # transposed contextual
        _ctx_rp_bucket_flatten = None
        if self.mode == 'contextual' and self.transposed:
            if RPEIndexFunction is None:
                offset = torch.arange(0, L * self.num_buckets, self.num_buckets,
                                      dtype=rp_bucket.dtype, device=rp_bucket.device).view(-1, 1)
                _ctx_rp_bucket_flatten = (rp_bucket + offset).flatten()
        self._rp_bucket_buf = (key, rp_bucket, _ctx_rp_bucket_flatten)
        return rp_bucket, _ctx_rp_bucket_flatten

    def forward_rpe_transpose(self, x, rp_bucket):
        """Forward function for iRPE (transposed version)
        This version is utilized by RPE on Query or Key

        Parameters
        ----------
        x: torch.Tensor
            Input Tensor whose shape is (B, H, L, head_dim),
            where B is batch size,
                  H is the number of heads,
                  L is the sequence length,
                    equal to height * width (+1 if class token exists)
                  head_dim is the dimension of each head
        rp_bucket: torch.Tensor
            relative position encoding buckets IDs
            The shape is (L, L)

        Weights
        -------
        lookup_table_bias: torch.Tensor
            The shape is (H or 1, num_buckets)

        or

        lookup_table_weight: torch.Tensor
            The shape is (H or 1, head_dim, num_buckets)

        Returns
        -------
        output: torch.Tensor
            Relative position encoding on queries or keys.
            The shape is (B or 1, H, L, L),
            where D is the output dimension for each head.
        """

        B = len(x)  # batch_size
        L_query, L_key = rp_bucket.shape
        if self.mode == 'bias':
            return self.lookup_table_bias[:, rp_bucket.flatten()].\
                view(1, self.num_heads, L_query, L_key)

        elif self.mode == 'contextual':
            """
            ret[b, h, i, j] = lookup_table_weight[b, h, i, rp_bucket[i, j]]

            ret[b, h, i * L_key + j] = \
               lookup_table[b, h, i * num_buckets + rp_buckets[i, j]]

            computational cost
            ------------------
            matmul: B * H * L_query * head_dim * num_buckets
            index: L_query + L_query * L_key + B * H * L_query * L_key
            total: O(B * H * L_query * (head_dim * num_buckets + L_key))
            """
            lookup_table = torch.matmul(
                x.transpose(0, 1).reshape(-1, B * L_query, self.head_dim),
                self.lookup_table_weight).\
                view(-1, B, L_query, self.num_buckets).transpose(0, 1)
            if RPEIndexFunction is not None:
                return RPEIndexFunction.apply(lookup_table, rp_bucket)
            else:
                return lookup_table.flatten(2)[:, :, self._ctx_rp_bucket_flatten].\
                    view(B, -1, L_query, L_key)

    def forward_rpe_no_transpose(self, x, rp_bucket):
        """Forward function for iRPE (non-transposed version)
        This version is utilized by RPE on Value.

        Parameters
        ----------
        x: torch.Tensor
            Input Tensor whose shape is (B, H, L, head_dim),
            where B is batch size,
                  H is the number of heads,
                  L is the sequence length,
                    equal to height * width (+1 if class token exists)
                  head_dim is the dimension of each head
        rp_bucket: torch.Tensor
            relative position encoding buckets IDs
            The shape is (L, L)

        Weights
        -------
        lookup_table_weight: torch.Tensor
            The shape is (H or 1, num_buckets, head_dim)

        Returns
        -------
        output: torch.Tensor
            Relative position encoding on values.
            The shape is (B, H, L, D),
            where D is the output dimension for each head.
        """

        B = len(x)  # batch_size
        L_query, L_key = rp_bucket.shape
        assert self.mode == 'contextual', "Only support contextual \
version in non-transposed version"
        weight = self.lookup_table_weight[:, rp_bucket.flatten()].\
            view(self.num_heads, L_query, L_key, self.head_dim)
        # (H, L_query, B, L_key) @ (H, L_query, L_key, D) = (H, L_query, B, D)
        # -> (B, H, L_query, D)
        return torch.matmul(x.permute(1, 2, 0, 3), weight).permute(2, 0, 1, 3)

    def __repr__(self):
        return 'iRPE(head_dim={rpe.head_dim}, num_heads={rpe.num_heads}, \
mode="{rpe.mode}", method={rpe.method}, transposed={rpe.transposed}, \
num_buckets={rpe.num_buckets}, initializer={rpe.initializer}, \
rpe_config={rpe.rpe_config})'.format(rpe=self)


class iRPE_Cross(nn.Module):
    """The implementation of image relative position encoding (specific for Cross method).

    Parameters
    ----------
    head_dim: int
        The dimension for each head.
    num_heads: int
        The number of parallel attention heads.
    mode: str or None
        The mode of image relative position encoding.
        Choices: [None, 'bias', 'contextual']
    method: METHOD
        The method ID of image relative position encoding.
        The `METHOD` class is defined in `irpe.py`.
    transposed: bool
        Whether to transpose the input feature.
        For iRPE on queries or keys, transposed should be `True`.
        For iRPE on values, transposed should be `False`.
    num_buckets: int
        The number of buckets, which store encodings.
    initializer: None or an inplace function
        [Optional] The initializer to `lookup_table`.
        Initalize `lookup_table` as zero by default.
    rpe_config: RPEConfig
        The config generated by the function `get_single_rpe_config`.
    """

    def __init__(self, method, **kwargs):
        super().__init__()
        assert method == METHOD.CROSS
        self.rp_rows = iRPE(**kwargs, method=METHOD.CROSS_ROWS)
        self.rp_cols = iRPE(**kwargs, method=METHOD.CROSS_COLS)

    def forward(self, x, height=None, width=None):
        """forward function for iRPE.
        Compute encoding on horizontal and vertical directions separately,
        then summarize them.

        Parameters
        ----------
        x: torch.Tensor
            Input Tensor whose shape is (B, H, L, head_dim),
            where B is batch size,
                  H is the number of heads,
                  L is the sequence length,
                    equal to height * width (+1 if class token exists)
                  head_dim is the dimension of each head
        height: int or None
            [Optional] The height of the input
            If not defined, height = floor(sqrt(L))
        width: int or None
            [Optional] The width of the input
            If not defined, width = floor(sqrt(L))

        Returns
        -------
        rpe_encoding: torch.Tensor
            Image Relative Position Encoding,
            whose shape is (B, H, L, L)
        """

        rows = self.rp_rows(x, height=height, width=width)
        cols = self.rp_cols(x, height=height, width=width)
        return rows + cols

    def __repr__(self):
        return 'iRPE_Cross(head_dim={rpe.head_dim}, \
num_heads={rpe.num_heads}, mode="{rpe.mode}", method={rpe.method}, \
transposed={rpe.transposed}, num_buckets={rpe.num_buckets}, \
initializer={rpe.initializer}, \
rpe_config={rpe.rpe_config})'.format(rpe=self.rp_rows)


def get_single_rpe_config(ratio=1.9,
                          method=METHOD.PRODUCT,
                          mode='contextual',
                          shared_head=True,
                          skip=0):
    """Get the config of single relative position encoding

    Parameters
    ----------
    ratio: float
        The ratio to control the number of buckets.
    method: METHOD
        The method ID of image relative position encoding.
        The `METHOD` class is defined in `irpe.py`.
    mode: str or None
        The mode of image relative position encoding.
        Choices: [None, 'bias', 'contextual']
    shared_head: bool
        Whether to share weight among different heads.
    skip: int
        The number of skip token before spatial tokens.
        When skip is 0, no classification token.
        When skip is 1, there is a classification token before spatial tokens.
        When skip > 1, there are `skip` extra tokens before spatial tokens.

    Returns
    -------
    config: RPEConfig
        The config of single relative position encoding.
    """
    config = edict()
    # whether to share encodings across different heads
    config.shared_head = shared_head
    # mode: None, bias, contextual
    config.mode = mode
    # method: None, Bias, Quant, Cross, Product
    config.method = method
    # the coefficients of piecewise index function
    config.alpha = 1 * ratio
    config.beta = 2 * ratio
    config.gamma = 8 * ratio

    # set the number of buckets
    config.num_buckets = get_num_buckets(method,
                                         config.alpha,
                                         config.beta,
                                         config.gamma)
    # add extra bucket for `skip` token (e.g. class token)
    if skip > 0:
        config.num_buckets += 1
    return config


def get_rpe_config(ratio=1.9,
                   method=METHOD.PRODUCT,
                   mode='contextual',
                   shared_head=True,
                   skip=0,
                   rpe_on='k'):
    """Get the config of relative position encoding on queries, keys and values

    Parameters
    ----------
    ratio: float
        The ratio to control the number of buckets.
    method: METHOD or str
        The method ID (or name) of image relative position encoding.
        The `METHOD` class is defined in `irpe.py`.
    mode: str or None
        The mode of image relative position encoding.
        Choices: [None, 'bias', 'contextual']
    shared_head: bool
        Whether to share weight among different heads.
    skip: int
        The number of skip token before spatial tokens.
        When skip is 0, no classification token.
        When skip is 1, there is a classification token before spatial tokens.
        When skip > 1, there are `skip` extra tokens before spatial tokens.
    rpe_on: str
        Where RPE attaches.
        "q": RPE on queries
        "k": RPE on keys
        "v": RPE on values
        "qk": RPE on queries and keys
        "qkv": RPE on queries, keys and values

    Returns
    -------
    config: RPEConfigs
        config.rpe_q: the config of relative position encoding on queries
        config.rpe_k: the config of relative position encoding on keys
        config.rpe_v: the config of relative position encoding on values
    """

    # alias
    if isinstance(method, str):
        method_mapping = dict(
            euc=METHOD.EUCLIDEAN,
            quant=METHOD.QUANT,
            cross=METHOD.CROSS,
            product=METHOD.PRODUCT,
        )
        method = method_mapping[method.lower()]
    if mode == 'ctx':
        mode = 'contextual'
    config = edict()
    # relative position encoding on queries, keys and values
    kwargs = dict(
        ratio=ratio,
        method=method,
        mode=mode,
        shared_head=shared_head,
        skip=skip,
    )
    config.rpe_q = get_single_rpe_config(**kwargs) if 'q' in rpe_on else None
    config.rpe_k = get_single_rpe_config(**kwargs) if 'k' in rpe_on else None
    config.rpe_v = get_single_rpe_config(**kwargs) if 'v' in rpe_on else None
    return config


def build_rpe(config, head_dim, num_heads):
    """Build iRPE modules on queries, keys and values.

    Parameters
    ----------
    config: RPEConfigs
        config.rpe_q: the config of relative position encoding on queries
        config.rpe_k: the config of relative position encoding on keys
        config.rpe_v: the config of relative position encoding on values
        None when RPE is not used.
    head_dim: int
        The dimension for each head.
    num_heads: int
        The number of parallel attention heads.

    Returns
    -------
    modules: a list of nn.Module
        The iRPE Modules on [queries, keys, values].
        None when RPE is not used.
    """
    if config is None:
        return None, None, None
    rpes = [config.rpe_q, config.rpe_k, config.rpe_v]
    transposeds = [True, True, False]

    def _build_single_rpe(rpe, transposed):
        if rpe is None:
            return None

        rpe_cls = iRPE if rpe.method != METHOD.CROSS else iRPE_Cross
        return rpe_cls(
            head_dim=head_dim,
            num_heads=1 if rpe.shared_head else num_heads,
            mode=rpe.mode,
            method=rpe.method,
            transposed=transposed,
            num_buckets=rpe.num_buckets,
            rpe_config=rpe,
        )
    return [_build_single_rpe(rpe, transposed)
            for rpe, transposed in zip(rpes, transposeds)]


if __name__ == '__main__':
    config = get_rpe_config(skip=1)
    rpe = build_rpe(config, head_dim=32, num_heads=4)
    print(rpe)
