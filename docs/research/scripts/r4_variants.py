"""Варианты бэкбона/FPN MyNet-1d для ЭКГ: страйды, параметры, MACs, рецептивное поле (R4)."""
import sys, types, torch, torch.nn as nn
import os; ROOT=os.path.abspath(os.path.join(os.path.dirname(__file__),'..','..','..')); sys.path.insert(0, os.path.join(ROOT,'src'))
# nn.py импортирует util (torchvision и пр.) только ради make_anchors -> подменяем лёгким модулем
import importlib.util
util = types.ModuleType('yolo_1d.utils.util')
def make_anchors(x, strides, offset=0.5):
    a,s=[],[]
    for i,st in enumerate(strides):
        w=x[i].shape[-1]; a.append(torch.arange(w,dtype=x[i].dtype)[:,None]+offset); s.append(torch.full((w,1),float(st)))
    return torch.cat(a),torch.cat(s)
util.make_anchors=make_anchors
sys.modules['yolo_1d']=types.ModuleType('yolo_1d'); sys.modules['yolo_1d.utils']=types.ModuleType('yolo_1d.utils')
sys.modules['yolo_1d.utils.util']=util
spec=importlib.util.spec_from_file_location('yolo_1d.utils.constants',os.path.join(ROOT,'src/yolo_1d/utils/constants.py')); c=importlib.util.module_from_spec(spec); spec.loader.exec_module(c); sys.modules['yolo_1d.utils.constants']=c
spec=importlib.util.spec_from_file_location('ynn',os.path.join(ROOT,'src/yolo_1d/nets/nn.py')); ynn=importlib.util.module_from_spec(spec); spec.loader.exec_module(ynn)
Conv,CSP,SPP,PSA,Head=ynn.Conv,ynn.CSP,ynn.SPP,ynn.PSA,ynn.Head
S=nn.SiLU

class Backbone(nn.Module):
    """Повторяет DarkNet, но каждая свёртка 'down' имеет настраиваемый шаг. s = список из 8 шагов (для 8 Conv)."""
    def __init__(self, cin, w, d, csp, s, attn=True):
        super().__init__()
        c=lambda i,o,st: Conv(i,o,S(),3,st,1)
        self.p1=nn.Sequential(c(cin,w[0],s[0]),c(w[0],w[1],s[1]),c(w[1],w[2],s[2]),CSP(w[2],w[2],d[0],csp[0],2),
                              c(w[2],w[3],s[3]),c(w[3],w[4],s[4]),CSP(w[4],w[4],d[1],csp[0],2))
        self.p2=nn.Sequential(c(w[4],w[5],s[5]),c(w[5],w[6],s[6]),CSP(w[6],w[6],d[2],csp[0],2))
        p3=[c(w[6],w[7],s[7]),CSP(w[7],w[7],d[3],csp[0],2),SPP(w[7],w[7])]
        if attn: p3.append(PSA(w[7],d[4]))
        self.p3=nn.Sequential(*p3)
    def forward(self,x):
        a=self.p1(x); b=self.p2(a); return a,b,self.p3(b)

class FPN(nn.Module):
    """DarkFPN с параметризованными отношениями страйдов r12=s2/s1, r23=s3/s2."""
    def __init__(self, w, d, csp, r12, r23):
        super().__init__()
        self.up1=nn.Upsample(scale_factor=r23); self.up2=nn.Upsample(scale_factor=r12)
        self.h1=CSP(w[6]+w[7],w[7],d[4],csp[1],2); self.h2=CSP(w[4]+w[7],w[7],d[5],csp[1],2)
        down=lambda r: [Conv(w[7],w[7],S(),3,2,1) for _ in range({1:0,2:1,4:2,8:3}[r])] or [nn.Identity()]
        self.h3=nn.Sequential(*down(r12)); self.h4=CSP(w[7]*2,w[7],d[6],csp[1],2)
        self.h5=nn.Sequential(*down(r23)); self.h6=CSP(w[7]*2,w[7],d[7],csp[2],2)
    def forward(self,x):
        p3,p4,p5=x
        p4=self.h1(torch.cat([self.up1(p5),p4],1)); p3=self.h2(torch.cat([self.up2(p4),p3],1))
        p4=self.h4(torch.cat([self.h3(p3),p4],1)); p5=self.h6(torch.cat([self.h5(p4),p5],1))
        return p3,p4,p5

class Model(nn.Module):
    def __init__(self, cin, w, d, csp, s, L, attn=True):
        super().__init__()
        import numpy as np
        st=np.cumprod(s); s1,s2,s3=int(st[4]),int(st[6]),int(st[7])
        self.strides=(s1,s2,s3)
        self.net=Backbone(cin,w,d,csp,s,attn); self.fpn=FPN(w,d,csp,s2//s1,s3//s2)
        self.head=Head(3,(w[7],)*3); self.head.stride=torch.tensor([float(v) for v in self.strides])
    def forward(self,x): return self.head(list(self.fpn(self.net(x))))

NANO=dict(w=[32,32,32,64,128,128,128,128],d=[1]*9,csp=[False,False,True])
SMALL=dict(w=[32,32,64,96,128,128,192,192],d=[1]*9,csp=[True,True,True])
VARIANTS={
 'orig (49152)':      ([2,2,2,2,2,2,2,2],49152),
 'orig @2048':        ([2,2,2,2,2,2,2,2],2048),
 'A {4,16,32}':       ([2,2,1,1,1,2,2,2],2048),
 'B {8,16,32}':       ([2,2,2,1,1,1,2,2],2048),
 'C {4,8,16}':        ([2,1,1,1,2,1,2,2],2048),
 'D {2,8,16}':        ([2,1,1,1,1,2,2,2],2048),
}

def macs(model,x):
    tot=[0]
    def hook(m,i,o):
        if isinstance(m,nn.Conv1d): tot[0]+=o.numel()//o.shape[0]*m.in_channels//m.groups*m.kernel_size[0]
    hs=[m.register_forward_hook(hook) for m in model.modules() if isinstance(m,nn.Conv1d)]
    model(x); [h.remove() for h in hs]; return tot[0]

def rf(model,cin,L,level_outputs):
    """Теоретическое РП (без внимания): ширина области входа с ненулевым градиентом центрального выхода уровня."""
    res=[]
    for lvl in range(3):
        x=torch.randn(1,cin,L,requires_grad=True)
        out=level_outputs(model,x)[lvl]
        out[0,:,out.shape[-1]//2].sum().backward()
        g=x.grad.abs().sum(1)[0]; nz=(g>0).nonzero()
        res.append(int(nz.max()-nz.min()+1))
    return res

if __name__=='__main__' and '--rf' not in sys.argv:
    torch.manual_seed(0)
    for size_name,cfg in [('nano',NANO),('small',SMALL)]:
        print(f'===== {size_name}')
        for name,(s,L) in VARIANTS.items():
            for cin in ([1] if L==2048 else [30]):
                m=Model(cin,cfg['w'],cfg['d'],cfg['csp'],s,L).eval()
                x=torch.zeros(1,cin,L)
                with torch.no_grad(): outs=m.fpn(m.net(x))
                params=sum(p.numel() for p in m.parameters())
                with torch.no_grad(): mac=macs(m,x)
                # РП без внимания: модель без PSA, в train-режиме BN не мешает градиенту
                m2=Model(cin,cfg['w'],cfg['d'],cfg['csp'],s,L,attn=False).train()
                if L<=8192: r_bb=rf(m2,cin,L,lambda mm,xx: mm.net(xx)); r_fpn=rf(m2,cin,L,lambda mm,xx: mm.fpn(mm.net(xx)))
                else: r_bb=r_fpn=['-']*3
                print(f'{name:16s} cin={cin:2d} L={L:5d} strides={m.strides} lens={[o.shape[-1] for o in outs]} '
                      f'params={params/1e6:.2f}M MACs={mac/1e6:.0f}M RF_backbone={r_bb} RF_after_FPN(no attn)={r_fpn}')


# --- рецептивное поле (eval-режим, без PSA), запуск: python r4_variants.py --rf
if __name__=='__main__' and '--rf' in sys.argv:
    torch.manual_seed(0)
    Lrf=16384
    for name,(s,L) in VARIANTS.items():
        if name=='orig @2048': continue
        m=Model(1,NANO['w'],NANO['d'],NANO['csp'],s,Lrf,attn=False).eval()
        for p in m.parameters(): p.requires_grad_(False)
        r_bb=rf(m,1,Lrf,lambda mm,xx: mm.net(xx)); r_fpn=rf(m,1,Lrf,lambda mm,xx: mm.fpn(mm.net(xx)))
        print(f'{name:14s} strides={m.strides} RF backbone (samples)={r_bb} -> ms {[round(v*2) for v in r_bb]} | RF after FPN={r_fpn} -> ms {[round(v*2) for v in r_fpn]}')
