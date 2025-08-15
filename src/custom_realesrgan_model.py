import torch
from basicsr.models.srgan_model import SRGANModel
from basicsr.utils.registry import MODEL_REGISTRY
from collections import OrderedDict

@MODEL_REGISTRY.register()
class CustomRealESRGANModel(SRGANModel):
    def __init__(self, opt):
        super(CustomRealESRGANModel, self).__init__(opt)  # Gọi đầy đủ init của SRGANModel
        # Không cần thêm gì vì SRGANModel đã khởi tạo cri_pix, cri_perceptual, cri_gan dựa trên opt

    @torch.no_grad()
    def feed_data(self, data):
        self.lq = data['lq'].to(self.device)
        if 'gt' in data:
            self.gt = data['gt'].to(self.device)

    def optimize_parameters(self, current_iter):
        for p in self.net_d.parameters():
            p.requires_grad = False

        self.optimizer_g.zero_grad()
        self.output = self.net_g(self.lq)

        l_g_total = 0
        loss_dict = OrderedDict()

        # Pixel loss (nếu được định nghĩa trong opt)
        if hasattr(self, 'cri_pix') and self.cri_pix:
            l_g_pix = self.cri_pix(self.output, self.gt)
            l_g_total += l_g_pix
            loss_dict['l_g_pix'] = l_g_pix

        # Perceptual loss (nếu được định nghĩa trong opt)
        if hasattr(self, 'cri_perceptual') and self.cri_perceptual:
            l_g_percep, l_g_style = self.cri_perceptual(self.output, self.gt)
            if l_g_percep is not None:
                l_g_total += l_g_percep
                loss_dict['l_g_percep'] = l_g_percep
            if l_g_style is not None:
                l_g_total += l_g_style
                loss_dict['l_g_style'] = l_g_style

        # GAN loss (nếu được định nghĩa trong opt)
        if hasattr(self, 'cri_gan') and self.cri_gan:
            fake_g_pred = self.net_d(self.output)
            l_g_gan = self.cri_gan(fake_g_pred, True, is_disc=False)
            l_g_total += l_g_gan
            loss_dict['l_g_gan'] = l_g_gan

        if l_g_total != 0:  # Chỉ backward nếu có loss
            l_g_total.backward()
        self.optimizer_g.step()

        # Optimize net_d (nếu dùng GAN)
        if hasattr(self, 'cri_gan') and self.cri_gan:
            for p in self.net_d.parameters():
                p.requires_grad = True

            self.optimizer_d.zero_grad()
            real_d_pred = self.net_d(self.gt)
            l_d_real = self.cri_gan(real_d_pred, True, is_disc=True)
            loss_dict['l_d_real'] = l_d_real
            l_d_real.backward()

            fake_d_pred = self.net_d(self.output.detach())
            l_d_fake = self.cri_gan(fake_d_pred, False, is_disc=True)
            loss_dict['l_d_fake'] = l_d_fake
            l_d_fake.backward()

            self.optimizer_d.step()

        if self.ema_decay > 0:
            self.model_ema(decay=self.ema_decay)

        self.log_dict = self.reduce_loss_dict(loss_dict)