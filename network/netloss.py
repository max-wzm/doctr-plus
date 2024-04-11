import random
import torch
import torch.nn as nn
import torch.nn.functional as F

# from data_process.utils import numpy_unwarping


class TpsWarp(nn.Module):
    def __init__(self, s):
        super(TpsWarp, self).__init__()
        iy, ix = torch.meshgrid(torch.linspace(-1, 1, s), torch.linspace(-1, 1, s))
        self.gs = torch.stack((ix, iy), dim=2).reshape((1, -1, 2)).to("cuda")
        self.sz = s

    def forward(self, src, dst):
        # src and dst are B.n.2
        B, n, _ = src.size()
        # B.n.1.2
        delta = src.unsqueeze(2)
        delta = delta - delta.permute(0, 2, 1, 3)
        # B.n.n
        K = delta.norm(dim=3)
        # Rsq = torch.sum(delta**2, dim=3)
        # Rsq += torch.eye(n, device='cuda')
        # Rsq[Rsq == 0] = 1.
        # K = 0.5 * Rsq * torch.log(Rsq)
        # c = -150
        # K = torch.exp(c * Rsq)
        # K = torch.abs(Rsq - 0.5) - 0.5
        # WARNING: TORCH.SQRT HAS NAN GRAD AT 0
        # K = torch.sqrt(Rsq)
        # print(K)
        # K[torch.isnan(K)] = 0.
        P = torch.cat((torch.ones((B, n, 1), device="cuda"), src), 2)
        L = torch.cat((K, P), 2)
        t = torch.cat((P.permute(0, 2, 1), torch.zeros((B, 3, 3), device="cuda")), 2)
        L = torch.cat((L, t), 1)
        # LInv = L.inverse()
        # # wv is B.n+3.2
        # wv = torch.matmul(LInv, torch.cat((dst, torch.zeros((B, 3, 2), device='cuda')), 1))
        # the above implementation has stability problem near the boundaries
        wv = torch.linalg.solve(
            L, torch.cat((dst, torch.zeros((B, 3, 2), device="cuda")), 1)
        )

        # get the grid sampler
        s = self.gs.size(1)
        gs = self.gs
        delta = gs.unsqueeze(2)
        delta = delta - src.unsqueeze(1)
        K = delta.norm(dim=3)
        gs = gs.expand(B, -1, -1)
        P = torch.cat((torch.ones((B, s, 1), device="cuda"), gs), 2)
        L = torch.cat((K, P), 2)
        gs = torch.matmul(L, wv)
        return gs.reshape(B, self.sz, self.sz, 2).permute(0, 3, 1, 2)


class LocalLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def warp_diff_loss(self, pred_bm, pred_perturb_bm, perturb_fm, perturb_bm):
        loss_f = F.l1_loss(
            pred_bm.clamp(-1.0, 1.0),
            F.grid_sample(
                perturb_fm, pred_perturb_bm.permute(0, 2, 3, 1), align_corners=True
            ).detach().clamp(-1.0, 1.0),
        )
        loss_b = F.l1_loss(
            pred_perturb_bm.clamp(-1.0, 1.0),
            F.grid_sample(
                perturb_bm, pred_bm.permute(0, 2, 3, 1), align_corners=True
            ).detach().clamp(-1.0, 1.0),
        )
        loss = loss_f + loss_b
        return loss


class WarperUtil(nn.Module):
    def __init__(self, imsize):
        super().__init__()
        self.tpswarper = TpsWarp(imsize)
        self.s = imsize

    def perturb_warp(self, batch_size):
        """
        returns: bm, fm with shape (Bx2xHxW) in [-1, 1]

        """
        # B = dd.size(0)
        B = batch_size
        s = self.s
        # -0.2 to 0.2
        iy, ix = torch.meshgrid(torch.linspace(-1, 1, s), torch.linspace(-1, 1, s))
        t = torch.stack((ix, iy), dim=0).unsqueeze(0).to("cuda").expand(B, -1, -1, -1)

        tt = t.clone()

        nd = random.randint(0, 4)
        for ii in range(nd):
            # define deformation on bd
            pm = (torch.rand(B, 1) - 0.5) * 0.2
            ps = (torch.rand(B, 1) - 0.5) * 1.95
            pt = ps + pm
            pt = pt.clamp(-0.975, 0.975)
            # put it on one bd
            # [1, 1] or [-1, 1] or [-1, -1] etc
            a1 = (torch.rand(B, 2) > 0.5).float() * 2 - 1
            # select one col for every row
            a2 = torch.rand(B, 1) > 0.5
            a2 = torch.cat([a2, a2.bitwise_not()], dim=1)
            a3 = a1.clone()
            a3[a2] = ps.view(-1)
            ps = a3.clone()
            a3[a2] = pt.view(-1)
            pt = a3.clone()
            # 2 N 4
            bds = torch.stack(
                [
                    t[0, :, 1:-1, 0],
                    t[0, :, 1:-1, -1],
                    t[0, :, 0, 1:-1],
                    t[0, :, -1, 1:-1],
                ],
                dim=2,
            )

            pbd = a2.bitwise_not().float() * a1
            # id of boundary p is on
            pbd = torch.abs(0.5 * pbd[:, 0] + 2.5 * pbd[:, 1] + 0.5).long()
            # ids of other boundaries
            pbd = torch.stack([pbd + 1, pbd + 2, pbd + 3], dim=1) % 4
            # print(pbd)
            pbd = bds[..., pbd].permute(2, 0, 1, 3).reshape(B, 2, -1)

            srcpts = torch.stack(
                [
                    t[..., 0, 0],
                    t[..., 0, -1],
                    t[..., -1, 0],
                    t[..., -1, -1],
                    ps.to("cuda"),
                ],
                dim=2,
            )
            srcpts = torch.cat([pbd, srcpts], dim=2).permute(0, 2, 1)
            dstpts = torch.stack(
                [
                    t[..., 0, 0],
                    t[..., 0, -1],
                    t[..., -1, 0],
                    t[..., -1, -1],
                    pt.to("cuda"),
                ],
                dim=2,
            )
            dstpts = torch.cat([pbd, dstpts], dim=2).permute(0, 2, 1)
            # print(srcpts)
            # print(dstpts)
            tgs = self.tpswarper(srcpts, dstpts)
            tt = F.grid_sample(tt, tgs.permute(0, 2, 3, 1), align_corners=True)

        nd = random.randint(1, 5)
        for ii in range(nd):

            pm = (torch.rand(B, 2) - 0.5) * 0.2
            ps = (torch.rand(B, 2) - 0.5) * 1.95
            pt = ps + pm
            pt = pt.clamp(-0.975, 0.975)

            srcpts = torch.cat(
                [
                    t[..., -1, :],
                    t[..., 0, :],
                    t[..., 1:-1, 0],
                    t[..., 1:-1, -1],
                    ps.unsqueeze(2).to("cuda"),
                ],
                dim=2,
            ).permute(0, 2, 1)
            dstpts = torch.cat(
                [
                    t[..., -1, :],
                    t[..., 0, :],
                    t[..., 1:-1, 0],
                    t[..., 1:-1, -1],
                    pt.unsqueeze(2).to("cuda"),
                ],
                dim=2,
            ).permute(0, 2, 1)
            tgs = self.tpswarper(srcpts, dstpts)
            tt = F.grid_sample(tt, tgs.permute(0, 2, 3, 1), align_corners=True)
        tgs = tt

        # sample tgs to gen invtgs
        num_sample = 512
        # n = (H-2)*(W-2)
        n = s * s
        idx = torch.randperm(n)
        idx = idx[:num_sample]
        srcpts = tgs.reshape(-1, 2, n)[..., idx].permute(0, 2, 1)
        dstpts = t.reshape(-1, 2, n)[..., idx].permute(0, 2, 1)
        invtgs = self.tpswarper(srcpts, dstpts)
        return tgs, invtgs


if __name__ == "__main__":
    warp = WarperUtil(64)
    bm, fm = warp.perturb_warp(1)
    print(bm.shape, bm.max(), bm.min())
    import cv2

    img = cv2.imread(
        "/data/home/mackswang/doc_rect/doctr-plus/data/QBdoc2/img/0a1a889d311046a28dc5cc06aecd58e1.jpg"
    )
    img = img.transpose(2, 0, 1)
    bm = bm.detach().cpu().numpy()
