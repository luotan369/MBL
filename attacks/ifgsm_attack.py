# https://github.com/fangshuman/transfer-attack-framework/blob/main/attacks/sgm.py
import numpy as np
import scipy.stats as st
import torch
import torch.nn.functional as F


def _batch_multiply_tensor_by_vector(vector, batch_tensor):
    return (batch_tensor.transpose(0, -1) * vector).transpose(0, -1).contiguous()


def batch_multiply(float_or_vector, tensor):
    if isinstance(float_or_vector, torch.Tensor):
        assert len(float_or_vector) == len(tensor)
        tensor = _batch_multiply_tensor_by_vector(float_or_vector, tensor)
    elif isinstance(float_or_vector, float):
        tensor *= float_or_vector
    else:
        raise TypeError("Value has to be float or torch.Tensor")
    return tensor


def _get_norm_batch(x, p):
    # 范数计算公式，给出张量以及p范数的值，计算张量x各自p范数的值
    batch_size = x.size(0)
    return x.abs().pow(p).view(batch_size, -1).sum(dim=1).pow(1. / p)


def normalize_by_pnorm(x, p=2, small_constant=1e-6):
    # 它用于将输入张量 x 按照 p-范数进行归一化
    assert isinstance(p, float) or isinstance(p, int)  # 确保p是指定类型的实例
    norm = _get_norm_batch(x, p)
    norm = torch.max(norm, torch.ones_like(norm) * small_constant)
    # 确保计算出的norm不为0（保证不出现除以0的情况），如果出现0用一个很小的常数来替代
    # torch.ones_like（x） 生成一个形状与x一致的张量其中的值全部以1替代
    return batch_multiply(1. / norm, x)  # 返回 x除以各自p范数后的值（即正则化）
    # 如果是1范数归一化，那么归一化后，张量中每个元素的绝对值之和就为1
    # 2范数，每个元素的平方和为1


class IFGSM_Based_Attacker(object):
    def __init__(
            self,
            attack_method,
            surrogate_model,
            args,
            dataset,
            linf_limit=0.05,
            iteration=10,
    ):
        self.attack_method = attack_method
        self.surrogate_model = surrogate_model
        self.loss_fn = F.cross_entropy
        print(f'eps: {linf_limit}')
        print(f'IFGSM Transfer Methods: {attack_method}-fgsm.')
        decay_factor = 1 if dataset == 'cifar10' else 0.2  # 梯度衰减因子 cifar为1表示不衰减 imagenet为0.2指梯度衰减时梯度乘以0.2
        if dataset == 'mnist':
            decay_factor = 1
        # decay_factor = 1 可以基于它来控制每次迭代梯度趋势的大小
        # eps_iter = 0.005  #  else linf_limit / iteration
        if dataset == 'imagenet':
            eps_iter = 0.005
        else:
            eps_iter = linf_limit / iteration
        default_value = {
            # basic default value
            'eps': linf_limit,    # 攻击扰动的最大强度
            'nb_iter': iteration,  # 迭代次数 imagenet 0.05 50次 步长0.01需要调整
            'eps_iter': eps_iter,  # 每次迭代扰动的步长
            'target': False,
            # extra default value
            'prob': 0.5,  # 攻击算法中随机性有关的概率值
            'kernlen': 7,  # kernel长度 可能用于滤波或者平滑操作
            'nsig': 3,  # nsig 与高斯噪声有关参数
            'decay_factor': decay_factor,
            'scale_copies': 5,  # 缩放副本数
            'sample_n': 20,  # 采样数量
            'sample_beta': 1.5,  # 可能与采样形状参数有关
            'amplification': 10,  # 放大因子，可能用于调整扰动强度
        }
        print(default_value)
        for k, v in default_value.items():
            self.load_params(k, v, args)

    def load_params(self, key, value, args):
        try:
            self.__dict__[key] = vars(args)[key]
        except:
            self.__dict__[key] = value

    def attack(self, model, image, label):
        # 生成对抗样本，返回成功与否
        model.eval()
        adv = self.perturb(image, label)  # 生成初步的对抗样本（每个元素的更新方向确定，但扰动还不是最大）

        delta = (adv - image).sign() * self.eps  # 获得扰动更新方向，并乘上扰动的最大强度（乘最大强度的原因可能是为了保证攻击成功率，因为方向基本上就是让loss最快变得更大的方向）
        adv = (image + delta).clamp(0., 1.)  # 将新获得的扰动加在图像上，并限制像素值为0到1保证图像有效
        # import pdb;pdb.set_trace()

        prob_output = torch.nn.functional.softmax(model(adv), dim=1)
        pred_lable = torch.argmax(prob_output, dim=1)
        pred_label = pred_lable.item()
        # 测试是否是成功的对抗样本并返回结果

        success = pred_label != label
        query_cnt = 1
        return success, query_cnt, adv, None

    def perturb(self, x, y):
        SEED = 0
        torch.manual_seed(SEED)
        torch.cuda.manual_seed(SEED)
        np.random.seed(SEED)

        eps_iter = self.eps_iter

        # initialize extra var
        if "mi" in self.attack_method or "ni" in self.attack_method:
            g = torch.zeros_like(x)  # g表示梯度形状与x张量相同
        if "vi" in self.attack_method:
            variance = torch.zeros_like(x)
        if "pi" in self.attack_method:
            a = torch.zeros_like(x)
            eps_iter *= self.amplification
            stack_kern, kern_size = self.project_kern(5)

        extra_item = torch.zeros_like(x)  # 和原始图像形状一样的0张量
        delta = torch.zeros_like(x)  # 这里的delta 应该就是代表扰动δ
        delta.requires_grad_()  # 计算梯度时会在delta对象的基础上进行修改

        for i in range(self.nb_iter):
        # 按照30次的迭代次数来生成对抗样本扰动delta delta的张量形状与图像x一致
        # 初步生成的图像不受最大扰动强度的限制，可能会导致扰动过大，图像质量不高的问题
        # 所以在attack阶段会调用pertub函数用这个不受限的对抗样本根据扰动强度生成受限的对抗样本。
            if "ni" in self.attack_method:
                img_x = x + self.decay_factor * eps_iter * g
            else:
                img_x = x

            # get gradient
            if "si" in self.attack_method:
                grad = torch.zeros_like(img_x)
                for i in range(self.scale_copies):
                    if "di" in self.attack_method:
                        outputs = self.surrogate_model(self.input_diversity(img_x + delta) * (1. / pow(2, i)))
                    else:
                        outputs = self.surrogate_model((img_x + delta) * (1. / pow(2, i)))

                    loss = self.loss_fn(outputs, y)
                    if self.target:
                        loss = -loss

                    loss.backward()
                    grad += delta.grad.data
                    delta.grad.data.zero_()
                # get average value of gradient
                grad = grad / self.scale_copies
            else:
                if "di" in self.attack_method:
                    outputs = self.surrogate_model(self.input_diversity(img_x + delta))
                else:
                    outputs = self.surrogate_model(img_x + delta)
                # 这里直接拿加上原图像扰动后代理模型的推理结果与目标标签计算交叉熵损失
                # 如果是有目标的攻击的话，算损失的负数。
                # 然后计算出让损失更大的梯度（包含方向和大小） ， 这里的损失是，原始图像加上新迭代得到的扰动之后图像 与 原图像推理结果的交叉熵损失函数
                loss = self.loss_fn(outputs, y)
                if self.target:
                    loss = -loss

                loss.backward()
                grad = delta.grad.data

            # variance: VI-FGSM
            if "vi" in self.attack_method:
                global_grad = torch.zeros_like(img_x)
                for i in range(self.sample_n):
                    r = torch.rand_like(img_x) * self.sample_beta * self.eps
                    r.requires_grad_()

                    outputs = self.surrogate_model(img_x + delta + r)

                    loss = self.loss_fn(outputs, y)
                    if self.target:
                        loss = -loss

                    loss.backward()
                    global_grad += r.grad.data
                    r.grad.data.zero_()

                current_grad = grad + variance

                # update variance
                variance = global_grad / self.sample_n - grad

                # return current_grad
                grad = current_grad

            # Gaussian kernel: TI-FGSM
            if "ti" in self.attack_method:
                kernel = self.get_Gaussian_kernel(img_x)
                grad = F.conv2d(grad, kernel, padding=self.kernlen // 2)

            # momentum: MI-FGSM / NI-FGSM
            if "mi" in self.attack_method or "ni" in self.attack_method:
                # g = self.decay_factor * g + torch.sign(grad)
                # 这里是按动量的方法，累计计算梯度。上一轮迭代的梯度值乘以衰减因子，加上当前范数1正则化后的梯度
                g = self.decay_factor * g + normalize_by_pnorm(grad, p=1)
                grad = g

            # Patch-wise attach: PI-FGSM
            if "pi" in self.attack_method:
                a += eps_iter * grad.data.sign()
                cut_noise = torch.clamp(abs(a) - self.eps, 0, 1e5) * a.sign()
                projection = eps_iter * (self.project_noise(cut_noise, stack_kern, kern_size)).sign()
                a += projection
                extra_item = projection  # return extra item

            grad_sign = grad.data.sign()
            delta.data = delta.data + eps_iter * grad_sign + extra_item
            # 每次迭代用上一轮的扰动加上本轮扰动方向乘以步长，得到新的扰动
            delta.data = torch.clamp(delta.data, -self.eps, self.eps)
            # 然后将扰动限制在最大扰动强度范围内，imagenet（-0.05,0.05） cifar-10(-8/255,8/255)
            delta.data = torch.clamp(x.data + delta, 0., 1.) - x
            # 保证加上扰动后图像的像素范围为0到1，再减去原图像得到最终的对抗性扰动
            delta.grad.data.zero_()  # 梯度置零避免梯度累积

        x_adv = torch.clamp(x + delta, 0.0, 1.0)
        # 将扰动加回原始图像得到对抗样本并返回（多次迭代后，每个元素的扰动方向确定）
        return x_adv

    def input_diversity(self, img):
        size = img.size(2)
        resize = int(size / 0.875)

        gg = torch.rand(1).item()
        if gg >= self.prob:
            return img
        else:
            rnd = torch.randint(size, resize + 1, (1,)).item()
            rescaled = F.interpolate(img, (rnd, rnd), mode="nearest")
            h_rem = resize - rnd
            w_hem = resize - rnd
            pad_top = torch.randint(0, h_rem + 1, (1,)).item()
            pad_bottom = h_rem - pad_top
            pad_left = torch.randint(0, w_hem + 1, (1,)).item()
            pad_right = w_hem - pad_left
            padded = F.pad(rescaled, pad=(pad_left, pad_right, pad_top, pad_bottom))
            padded = F.interpolate(padded, (size, size), mode="nearest")
            return padded

    def get_Gaussian_kernel(self, x):
        # define Gaussian kernel
        kern1d = st.norm.pdf(np.linspace(-self.nsig, self.nsig, self.kernlen))
        kernel = np.outer(kern1d, kern1d)
        kernel = kernel / kernel.sum()
        kernel = torch.FloatTensor(kernel).expand(x.size(1), x.size(1), self.kernlen, self.kernlen)
        kernel = kernel.to(x.device)
        return kernel

    def project_kern(self, kern_size):
        kern = np.ones((kern_size, kern_size), dtype=np.float32) / (kern_size ** 2 - 1)
        kern[kern_size // 2, kern_size // 2] = 0.0
        kern = kern.astype(np.float32)
        stack_kern = np.stack([kern, kern, kern])
        stack_kern = np.expand_dims(stack_kern, 1)
        stack_kern = torch.tensor(stack_kern).cuda()
        return stack_kern, kern_size // 2

    def project_noise(self, x, stack_kern, kern_size):
        x = F.conv2d(x, stack_kern, padding=(kern_size, kern_size), groups=3)
        return x
