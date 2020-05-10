import numpy as np

class Wavelet:
    def __init__(self, max_order, length, orthonormal=True, flexible=False):
        self.max_order = max_order
        self.length = length
        self.on = orthonormal
        self.flexible = flexible
        self.max_u = self._get_max_u
        self.family = {}
        self._generate_family()

    def _generate_family(self):
        pass

    def _get_max_u(self):
        pass

    def get_wavelet(self, s, u, approx=False):
        try:
            if approx:
                return self.family['father']
            return self.family[(s,u)]

        except KeyError:
            # TODO: Check the recalculations
            if s > self.max_order:
                if not self.flexible:
                    raise ValueError(
                        f"Order {s} is greater than defined max order {self.max_order}")
                else:
                    self.max_order = s
                    self._generate_family()
            elif u > self.max_u:
                if not self.flexible:
                    raise ValueError(
                        f"Time support {u} is greater than defined max time support {self.max_u}")
                else:
                    self.max_u = u
                    self._generate_family()

            return self.family[(s,u)]

    def get_xfrm_mat(self, s=None, u=None):
        pass

    def __call__(self, s, u):
        return self.get_wavelet(s, u)


class Haar(Wavelet):
    def __init__(self, max_order, length, orthonormal=True, flexible=False):
        super().__init__(max_order, length, orthonormal, flexible)

    def _get_max_u(self):
        self.max_u = 2 ** self.max_order

    def _generate_family(self):
        max_s = self.max_order
        if self.on:
            scale = 1 / np.sqrt(2 ** (max_s+1))
        else:
            scale = 1
        self.family['father'] = np.ones(self.length) * scale
        scale = 1
        for s in range(max_s + 1):
            for u in range(2 ** (max_s - s)):
                if self.on:
                    scale = 1 / np.sqrt(2 ** (s + 1))
                hlf_wavelet_len = 2 ** (s)
                subwvlt = np.concatenate([
                    np.ones(hlf_wavelet_len) * -1,
                    np.ones(hlf_wavelet_len)]) * scale
                shift = u * 2 * hlf_wavelet_len
                tail = self.length - (shift + 2 * hlf_wavelet_len)
                self.family[(s,u)] = np.concatenate([
                    np.zeros(shift),
                    subwvlt,
                    np.zeros(tail)
                ])

    def get_xfrm_mat(self, s=None, u=None):
        vectors = []
        vectors.append(self.family['father'])
        if not s and not u:
            for S in reversed(range(self.max_order + 1)):
                for U in range(2 ** (self.max_order - S)):
                    vectors.append(self.family[(S, U)])
        elif s:
            if not u:
                for S in range(s + 1):
                    for U in range(2 ** S):
                        vectors.append(self.family[(S, U)])
            else:
                for S in range(s + 1):
                    for U in range(u + 1):
                        vectors.append(self.family[(S, U)])

        return np.stack(vectors)