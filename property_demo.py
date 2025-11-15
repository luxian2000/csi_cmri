import torch

class QuantumState:
    """
    一个演示@property用法的量子态类
    """
    def __init__(self, alpha_real, alpha_imag, beta_real, beta_imag):
        """
        初始化量子态 |ψ> = α|0> + β|1>
        
        参数:
        alpha_real: |0>态的实部
        alpha_imag: |0>态的虚部
        beta_real: |1>态的实部
        beta_imag: |1>态的虚部
        """
        # 使用私有属性存储内部状态
        self._alpha_real = alpha_real
        self._alpha_imag = alpha_imag
        self._beta_real = beta_real
        self._beta_imag = beta_imag
    
    # 使用@property将方法转换为只读属性
    @property
    def alpha(self):
        """获取α系数"""
        return complex(self._alpha_real, self._alpha_imag)
    
    @property
    def beta(self):
        """获取β系数"""
        return complex(self._beta_real, self._beta_imag)
    
    @property
    def probability_alpha(self):
        """计算α的概率幅 (|α|²)"""
        return abs(self.alpha)**2
    
    @property
    def probability_beta(self):
        """计算β的概率幅 (|β|²)"""
        return abs(self.beta)**2
    
    # 使用setter允许修改属性
    @alpha.setter
    def alpha(self, value):
        """设置α系数"""
        if isinstance(value, complex):
            self._alpha_real = value.real
            self._alpha_imag = value.imag
        else:
            self._alpha_real = value
            self._alpha_imag = 0
    
    @beta.setter
    def beta(self, value):
        """设置β系数"""
        if isinstance(value, complex):
            self._beta_real = value.real
            self._beta_imag = value.imag
        else:
            self._beta_real = value
            self._beta_imag = 0
    
    @property
    def is_normalized(self):
        """检查量子态是否归一化"""
        total_prob = self.probability_alpha + self.probability_beta
        return abs(total_prob - 1.0) < 1e-10
    
    def normalize(self):
        """归一化量子态"""
        total_prob = self.probability_alpha + self.probability_beta
        norm_factor = torch.sqrt(torch.tensor(total_prob))
        
        self._alpha_real /= norm_factor.item()
        self._alpha_imag /= norm_factor.item()
        self._beta_real /= norm_factor.item()
        self._beta_imag /= norm_factor.item()
    
    def __str__(self):
        """字符串表示"""
        return f"α={self.alpha:.3f}, β={self.beta:.3f}"

# 演示@property的用法
if __name__ == "__main__":
    # 创建一个量子态 |ψ> = 0.6|0> + 0.8|1>
    state = QuantumState(0.6, 0.0, 0.8, 0.0)
    
    print("初始量子态:")
    print(f"  {state}")
    print(f"  |α|² = {state.probability_alpha:.3f}")
    print(f"  |β|² = {state.probability_beta:.3f}")
    print(f"  是否归一化: {state.is_normalized}")
    
    # 修改alpha值
    state.alpha = complex(0.5, 0.5)  # 设置为复数
    print("\n修改α系数后:")
    print(f"  {state}")
    print(f"  |α|² = {state.probability_alpha:.3f}")
    print(f"  |β|² = {state.probability_beta:.3f}")
    print(f"  是否归一化: {state.is_normalized}")
    
    # 归一化
    state.normalize()
    print("\n归一化后:")
    print(f"  {state}")
    print(f"  |α|² = {state.probability_alpha:.3f}")
    print(f"  |β|² = {state.probability_beta:.3f}")
    print(f"  总概率 = {state.probability_alpha + state.probability_beta:.3f}")
    print(f"  是否归一化: {state.is_normalized}")