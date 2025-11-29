import torch
import torch.nn as nn
from torch.autograd import grad
from contextlib import contextmanager

from problem import Stokes, DiffusionReaction


# ------------------------------------------------------------
# EMPTY CONTEXT MANAGER (αντί για paddle.no_grad)
# ------------------------------------------------------------
@contextmanager
def empty_context_manager():
    yield


# ------------------------------------------------------------
# GPU MEMORY PRINT (προαιρετικό)
# ------------------------------------------------------------
def print_max_memo():
    if torch.cuda.is_available():
        allocated_memory_bytes = torch.cuda.max_memory_allocated()
        allocated_memory_gb = allocated_memory_bytes / (1024 ** 3)
        print(f"Currently allocated GPU memory: {allocated_memory_gb:.2f} GB")
    else:
        print("CUDA not available.")


# ------------------------------------------------------------
# TILE (torch version του paddle.tile + concat)
# ------------------------------------------------------------
def tile(x: torch.Tensor, y: torch.Tensor):
    """
    Αντίστοιχο του paddle:
    X = x.tile([y.shape[0], 1])
    Y = concat(y[i].tile([x.shape[0],1]))
    return concat([X, Y], axis=1)
    """
    X = x.repeat(y.shape[0], 1)

    Y_list = []
    for i in range(y.shape[0]):
        Y_list.append(y[i].unsqueeze(0).repeat(x.shape[0], 1))

    Y = torch.cat(Y_list, dim=0)
    return torch.cat([X, Y], dim=1)


# ============================================================
# BSNN NETWORK (PyTorch)
# ============================================================
class BSNN(nn.Module):
    def __init__(self, layers, act="sin"):
        super(BSNN, self).__init__()

        self.layers = layers
        self.num_layers = len(layers)

        if act == "sin":
            self.act = torch.sin
        elif act == "relu":
            self.act = torch.relu
        else:
            raise NotImplementedError

        self.width = (
            [layers[0]]
            + [int(pow(2, i - 1) * layers[i]) for i in range(1, len(layers) - 1)]
            + [layers[-1]]
        )

        self.masks = self.construct_mask()

        self.layers_list = nn.ModuleList()
        for i in range(len(layers) - 1):
            in_features = self.width[i]
            out_features = self.width[i + 1]
            self.layers_list.append(nn.Linear(in_features, out_features))

    # --------------------------------------------------------
    # MASK CONSTRUCTION
    # --------------------------------------------------------
    def construct_mask(self):
        masks = []
        for l in range(2, self.num_layers - 2):
            num_blocks = int(pow(2, l - 1))
            blocksize1 = int(self.width[l] / num_blocks)
            blocksize2 = 2 * self.layers[l + 1]

            blocks = [torch.ones((blocksize1, blocksize2)) for _ in range(num_blocks)]
            mask = torch.block_diag(*blocks)
            masks.append(mask)

        return masks

    # --------------------------------------------------------
    # FORWARD
    # --------------------------------------------------------
    def forward(self, x):
        for i, layer in enumerate(self.layers_list):
            if 2 <= i < len(self.layers) - 2:
                W2 = layer.weight * self.masks[i - 2].to(layer.weight.device)
                x = torch.add(torch.matmul(x, W2.T), layer.bias)
                x = self.act(x)

            elif i == len(self.layers) - 2:
                x = layer(x)

            else:
                x = layer(x)
                x = self.act(x)

        return x


# ============================================================
# NET_INTEGRAL (PyTorch)
# ============================================================
class Net_Integral(nn.Module):
    def __init__(
        self,
        layers,
        shape,
        ngs_boundary,
        ngs_interior,
        problem,
        act="sin",
        eval_mode=False,
    ):
        super().__init__()

        self.G = []

        if len(shape) > 1:
            for i in range(shape[0]):
                Row = []
                for j in range(shape[1]):
                    Row.append(BSNN(layers[i][j], act))
                self.G.append(nn.ModuleList(Row))
            self.G = nn.ModuleList(self.G)
        else:
            raise NotImplementedError

        self.ngs_interior = ngs_interior
        self.ngs_boundary = ngs_boundary
        self.problem = problem

        if eval_mode is True:
            self.no_grad = torch.no_grad
        else:
            self.no_grad = empty_context_manager

    # --------------------------------------------------------
    # FORWARD
    # --------------------------------------------------------
    def forward(
        self,
        x_in_coord_list,
        x_in_wts_list,
        x_bc_wts_list,
        x_bc_coord_list,
        z,
        f_interior_list,
        g_boundary_list,
        a_boundary_list,
        x_bc_normal,
    ):

        N_interior = x_in_coord_list[0].shape[0]
        N_boundary = x_bc_coord_list[0].shape[0]

        fG_quad = torch.zeros_like(z)[:, 0]
        gGn_quad = torch.zeros_like(fG_quad)

        # ----------------------------------------------------
        def forward_layer(BSNN_interior, BSNN_boundary, fG_quad, gGn_quad):

            with self.no_grad():

                # ---------- INTERIOR ----------
                for k in range(self.ngs_interior):

                    INPUT_interior = tile(x_in_coord_list[k], z)
                    G_interior = BSNN_interior(INPUT_interior)

                    f_interior = f_interior_list[k]
                    f_interior = f_interior.repeat(z.shape[0], 1)

                    G_interior = (
                        G_interior.reshape(-1, N_interior).transpose(0, 1)
                    )
                    f_interior = (
                        f_interior.reshape(-1, N_interior).transpose(0, 1)
                    )

                    fG_interior = f_interior * G_interior
                    fG_quad += (fG_interior.transpose(0, 1) @ x_in_wts_list[k])

            # ---------- BOUNDARY ----------
            for k in range(self.ngs_boundary):

                INPUT_boundary = tile(x_bc_coord_list[k], z)
                INPUT_boundary.requires_grad_(True)

                G_boundary = BSNN_boundary(INPUT_boundary)
                G_boundary = (
                    G_boundary.reshape(-1, N_boundary).transpose(0, 1)
                )

                g_boundary = g_boundary_list[k]
                a_boundary = a_boundary_list[k]

                # -------- STOKES CASE --------
                if isinstance(self.problem, Stokes):

                    Ggrad = grad(
                        outputs=G_boundary,
                        inputs=INPUT_boundary,
                        grad_outputs=torch.ones_like(G_boundary),
                        create_graph=True,
                        retain_graph=True,
                    )[0]

                    Gx = Ggrad[:, 0].reshape(N_boundary, -1)
                    Gy = Ggrad[:, 1].reshape(N_boundary, -1)
                    Gz = Ggrad[:, 2].reshape(N_boundary, -1)

                    G_boundary = (
                        Gx * x_bc_normal[:, [0]]
                        + Gy * x_bc_normal[:, [1]]
                        + Gz * x_bc_normal[:, [2]]
                    )

                # -------- DIFFUSION CASE --------
                else:
                    g_boundary = g_boundary.repeat(z.shape[0], 1)
                    g_boundary = (
                        g_boundary.reshape(-1, N_boundary).transpose(0, 1)
                    )

                    if isinstance(self.problem, DiffusionReaction) and self.problem.geometry == "pipe":
                        a_boundary = 1.0
                    else:
                        a_boundary = a_boundary.repeat(z.shape[0], 1)
                        a_boundary = (
                            a_boundary.reshape(-1, N_boundary).transpose(0, 1)
                        )

                gGn_boundary = a_boundary * g_boundary * G_boundary
                gGn_quad += (gGn_boundary.transpose(0, 1) @ x_bc_wts_list[k])

            return fG_quad, gGn_quad

        # ----------------------------------------------------
        if isinstance(self.problem, Stokes):
            for j in range(len(self.G[0])):
                fG_quad, gGn_quad = forward_layer(
                    self.G[0][j], self.G[0][j], fG_quad, gGn_quad
                )
        else:
            fG_quad, gGn_quad = forward_layer(
                self.G[0][0], self.G[0][1], fG_quad, gGn_quad
            )

        quad_res = (fG_quad - gGn_quad)[:, None]
        return quad_res
