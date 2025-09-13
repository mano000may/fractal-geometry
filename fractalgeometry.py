import os
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "1"

import pygame
import numpy as np
import sys
import time
import math
from collections import deque
from types import SimpleNamespace

try:
    import taichi as ti
    TAICHI_AVAILABLE = True
except Exception:
    TAICHI_AVAILABLE = False
    
try:
    import mpmath
    MPMATH_AVAILABLE = True
except Exception:
    MPMATH_AVAILABLE = False


DEFAULT_WIDTH, DEFAULT_HEIGHT = 1024, 768
SIDEBAR_WIDTH = 320
FPS_LIMIT = 60
PAN_RENDER_FPS = 20
GPU_PRECISION = ti.f32 if TAICHI_AVAILABLE else None

FRACTAL_SETS = {
    0: "Mandelbrot", 1: "Julia", 2: "Burning Ship", 3: "Tricorn",
    4: "Mandelbar", 5: "Newton", 6: "Nova"
}
PALETTES = {0:"grayscale", 1:"classic", 2:"fire", 3:"ocean"}


def palette_map(val, scheme=1):
    v = np.clip(val, 0.0, 1.0)
    if scheme==0: p=(v*255).astype(np.uint8); return np.stack([p,p,p],axis=-1)
    if scheme==1: r=(np.sin(3.0+5.0*v)*0.5+0.5);g=(np.sin(1.0+5.0*v)*0.5+0.5);b=(np.sin(5.0+5.0*v)*0.5+0.5); return (np.stack([r,g,b],axis=-1)*255).astype(np.uint8)
    if scheme==2: r=np.clip(3.0*v,0,1);g=np.clip(3.0*(v-0.33),0,1);b=np.clip(3.0*(v-0.66),0,1); return (np.stack([r,g,b],axis=-1)*255).astype(np.uint8)
    if scheme==3: r=np.clip(2.0*(0.5-v),0,1);g=np.clip(np.sin(2.0*math.pi*v),-1,1)*0.5+0.5;b=np.clip(v**0.7,0,1); return (np.stack([r,g,b],axis=-1)*255).astype(np.uint8)
    p=(v*255).astype(np.uint8); return np.stack([p,p,p],axis=-1)

def render_fractal_numpy(width, height, center_x, center_y, span, max_iter, power,
                         c_re, c_im, fractal_set_idx=0, escape_radius=16.0,
                         supersample=1, palette=1, use_mpmath=False, precision=50,
                         nova_R=1.0, nova_C=(0.0, 0.0)):
    W,H=int(width*supersample),int(height*supersample); aspect=float(W)/float(H); view_h=span; view_w=span*aspect; xmin=center_x-view_w/2.0; ymin=center_y-view_h/2.0; xs=np.linspace(xmin,xmin+view_w,W,dtype=np.float64); ys=np.linspace(ymin,ymin+view_h,H,dtype=np.float64); X,Y=np.meshgrid(xs,ys); P=X+1j*Y
    M=np.full(P.shape,max_iter,dtype=np.int32)
    if fractal_set_idx in [5, 6]:
        ROOTS = [complex(1, 0), complex(-0.5, 0.8660254), complex(-0.5, -0.8660254)]; Z = P.copy(); nova_c_complex = complex(nova_C[0], nova_C[1])
        for k in range(max_iter):
            mask = M == max_iter;
            if not mask.any(): break
            Z_mask = Z[mask]; fz = Z_mask**3 - 1; fpz = 3 * Z_mask**2; fpz[np.abs(fpz) < 1e-9] = 1e-9
            if fractal_set_idx == 5: Z[mask] = Z_mask - fz / fpz
            else: Z[mask] = Z_mask - nova_R * fz / fpz + nova_c_complex
            for i, root in enumerate(ROOTS):
                converged_mask = (np.abs(Z[mask] - root) < 1e-3)
                M[mask][converged_mask] = k + (i * (max_iter // 3))
        norm = M / max_iter
    else:
        Z=np.zeros_like(P); C=P
        if fractal_set_idx==1: Z=P; C=np.full_like(P,complex(c_re,c_im))
        mask=np.ones(P.shape,dtype=bool)
        for k in range(max_iter):
            if not mask.any(): break
            if fractal_set_idx==2: Z[mask]=np.abs(Z[mask].real)+1j*np.abs(Z[mask].imag)
            elif fractal_set_idx==3: Z[mask]=np.conj(Z[mask])
            elif fractal_set_idx==4: Z[mask]=abs(Z[mask].real)+1j*Z[mask].imag
            Z[mask]=Z[mask]**power+C[mask]; escaped=np.abs(Z)>escape_radius; newly_escaped=escaped&mask; M[newly_escaped]=k; mask&=~newly_escaped
        absZ=np.abs(Z)
        with np.errstate(divide='ignore',invalid='ignore'):
            log_abs=np.log(absZ); nu=np.zeros_like(absZ,dtype=np.float64); escaped_mask=M<max_iter; denom=math.log(abs(power)) if abs(power)>0 else 1.0
            nu[escaped_mask]=M[escaped_mask]+1-np.log(np.clip(log_abs[escaped_mask],1e-16,np.inf))/denom; norm=np.clip(nu/(max_iter/2.0),0.0,1.0)
    rgb=palette_map(norm,scheme=palette)
    rgb[M>=max_iter]=0
    if supersample>1: 
        rgb=rgb.reshape((height,supersample,width,supersample,3)).mean(axis=(1,3)).astype(np.uint8)
    return np.transpose(rgb, (1, 0, 2))

if TAICHI_AVAILABLE:
    ti.init(arch=ti.gpu, default_fp=GPU_PRECISION)
    ROOTS_ti = ti.Vector.field(2, dtype=GPU_PRECISION, shape=3)
    ROOTS_ti[0] = [1.0, 0.0]; ROOTS_ti[1] = [-0.5, 0.8660254]; ROOTS_ti[2] = [-0.5, -0.8660254]

    @ti.func
    def complex_pow_ti(z_re,z_im,p):
        res_re,res_im=0.0,0.0; r=ti.sqrt(z_re*z_re+z_im*z_im)
        if r>0.0: theta=ti.atan2(z_im,z_re); rr=r**p; th=theta*p; res_re=rr*ti.cos(th); res_im=rr*ti.sin(th)
        return res_re,res_im
        
    @ti.func
    def palette_map_ti(v_norm,scheme:ti.i32):
        v=ti.min(1.0,ti.max(0.0,v_norm)); r,g,b=0.0,0.0,0.0
        if scheme==0: p=v; r,g,b=p,p,p
        elif scheme==1: r=ti.sin(3.0+5.0*v)*0.5+0.5; g=ti.sin(1.0+5.0*v)*0.5+0.5; b=ti.sin(5.0+5.0*v)*0.5+0.5
        elif scheme==2: r=ti.min(1.0,3.0*v); g=ti.min(1.0,ti.max(0.0,3.0*(v-0.33))); b=ti.min(1.0,ti.max(0.0,3.0*(v-0.66)))
        elif scheme==3: r=ti.min(1.0,ti.max(0.0,2.0*(0.5-v))); g=ti.sin(2.0*3.14159*v)*0.5+0.5; b=ti.min(1.0,v**0.7)
        else: r=ti.sin(3.0+5.0*v)*0.5+0.5; g=ti.sin(1.0+5.0*v)*0.5+0.5; b=ti.sin(5.0+5.0*v)*0.5+0.5
        r_u8=ti.cast(ti.min(1.0,ti.max(0.0,r))*255,ti.u8); g_u8=ti.cast(ti.min(1.0,ti.max(0.0,g))*255,ti.u8); b_u8=ti.cast(ti.min(1.0,ti.max(0.0,b))*255,ti.u8)
        return ti.Vector([r_u8,g_u8,b_u8])

    @ti.kernel
    def fractal_kernel_ti(pixels:ti.template(),fractal_width:ti.i32,height:ti.i32,fractal_set_idx:ti.i32,power:GPU_PRECISION,max_iter:ti.i32,c_re:GPU_PRECISION,c_im:GPU_PRECISION,cx:GPU_PRECISION,cy:GPU_PRECISION,span:GPU_PRECISION,escape_radius:GPU_PRECISION,palette_idx:ti.i32,nova_R:GPU_PRECISION,nova_C_re:GPU_PRECISION,nova_C_im:GPU_PRECISION):
        aspect=float(fractal_width)/float(height); view_w=span*aspect; view_h=span; xmin=cx-view_w/2.0; ymin=cy-view_h/2.0
        log_power=1.0;
        if power!=0.0: log_power=ti.log(abs(power))
        
        for i,j in pixels:
            x, y = xmin+(i+0.5)/fractal_width*view_w, ymin+(j+0.5)/height*view_h
            it = max_iter
            
            if fractal_set_idx == 5 or fractal_set_idx == 6:
                Z_re, Z_im = x, y
                for k in range(max_iter):
                    fz_re, fz_im = complex_pow_ti(Z_re, Z_im, 3); fz_re -= 1.0
                    fpz_re, fpz_im = complex_pow_ti(Z_re, Z_im, 2); fpz_re *= 3.0; fpz_im *= 3.0
                    den = fpz_re*fpz_re + fpz_im*fpz_im
                    if den < 1e-9: den = 1e-9
                    div_re = (fz_re*fpz_re + fz_im*fpz_im) / den; div_im = (fz_im*fpz_re - fz_re*fpz_im) / den
                    if fractal_set_idx == 5: Z_re -= div_re; Z_im -= div_im
                    else: Z_re -= nova_R * div_re - nova_C_re; Z_im -= nova_R * div_im - nova_C_im
                    for root_idx in ti.static(range(3)):
                        dist_sq = (Z_re - ROOTS_ti[root_idx][0])**2 + (Z_im - ROOTS_ti[root_idx][1])**2
                        if dist_sq < 1e-6 and it == max_iter:
                            it = k + (root_idx * (max_iter//3))
                    if it != max_iter:
                        break
                
                if it == max_iter: pixels[i,j].fill(ti.cast(0, ti.u8))
                else: pixels[i,j] = palette_map_ti(it/max_iter, palette_idx)
            
            else:
                Z_re,Z_im,C_re,C_im=0.0,0.0,x,y
                if fractal_set_idx==1: Z_re,Z_im=x,y; C_re,C_im=c_re,c_im
                for k in range(max_iter):
                    if Z_re*Z_re+Z_im*Z_im>escape_radius: it=k; break
                    z_re_temp,z_im_temp=Z_re,Z_im
                    if fractal_set_idx==2: z_re_temp=abs(Z_re); z_im_temp=abs(Z_im)
                    elif fractal_set_idx==3: z_im_temp=-Z_im
                    elif fractal_set_idx==4: z_re_temp=abs(Z_re)
                    Z_re,Z_im=complex_pow_ti(z_re_temp,z_im_temp,power); Z_re+=C_re; Z_im+=C_im
                
                if it==max_iter: pixels[i,j].fill(ti.cast(0, ti.u8))
                else: log_zn=0.5*ti.log(Z_re*Z_re+Z_im*Z_im); nu=float(it)+1.0-ti.log(log_zn)/log_power; norm=nu/(max_iter/2.0); pixels[i,j]=palette_map_ti(norm,palette_idx)


class Slider:
    def __init__(self,x,y,w,h,min_val,max_val,initial,label=""): self.rect=pygame.Rect(x,y,w,h); self.min_val=min_val; self.max_val=max_val; self.val=initial; self.dragging=False; self.label=label
    def handle_event(self,event):
        if event.type==pygame.MOUSEBUTTONDOWN and self.rect.collidepoint(event.pos): self.dragging=True
        elif event.type==pygame.MOUSEBUTTONUP and self.dragging: self.dragging=False; return True
        elif event.type==pygame.MOUSEMOTION and self.dragging: relx=(event.pos[0]-self.rect.x)/max(1,self.rect.w); self.val=self.min_val+relx*(self.max_val-self.min_val); self.val=max(self.min_val,min(self.max_val,self.val))
        return False
    def draw(self,surf,font):
        pygame.draw.rect(surf,(40,40,40),self.rect); pygame.draw.rect(surf,(120,120,120),self.rect,1); handle_x=int(self.rect.x+(self.val-self.min_val)/(self.max_val-self.min_val)*self.rect.w); pygame.draw.rect(surf,(200,200,220),(handle_x-6,self.rect.y-3,12,self.rect.h+6)); txt=f"{self.label}: {self.val:.3f}" if isinstance(self.val,float) else f"{self.label}: {int(self.val)}"; surf.blit(font.render(txt,True,(255,255,255)),(self.rect.x+self.rect.w+8,self.rect.y-4))

class FractalApp:
    def __init__(self):
        pygame.init()
        self.monitor_size=(pygame.display.Info().current_w,pygame.display.Info().current_h)
        self.default_window_size=(DEFAULT_WIDTH+SIDEBAR_WIDTH,DEFAULT_HEIGHT)
        self.fullscreen=False
        self.clock=pygame.time.Clock(); self.font=pygame.font.Font(None,20); self.font_big=pygame.font.Font(None,28)
        self.fractal_set_idx=0; self.center_x=-0.5; self.center_y=0.0; self.span=3.5; self.power=2.0; self.max_iter=200
        self.julia_c_re=-0.8; self.julia_c_im=0.156; self.escape_radius=16.0; self.supersample=1; self.palette=1
        self.nova_R=1.0; self.nova_C_re=0.0; self.nova_C_im=0.0
        self.auto_iter=True; self.mpmath_mode=False; self.mpmath_precision=50
        self.log_messages=deque(maxlen=20)
        self.needs_render=True; self.last_render_time=0.0; self.dragging=False; self.drag_start=(0,0); self.center_start=(self.center_x,self.center_y)
        self.last_pan_render_time = 0.0
        self.resize_window(*self.default_window_size)
        self.add_log("Application started.")

    def resize_window(self,width,height):
        self.width,self.height=width,height; self.fractal_width=self.width-SIDEBAR_WIDTH
        flags=pygame.FULLSCREEN if self.fullscreen else 0
        self.screen=pygame.display.set_mode((self.width,self.height),flags)
        self.fractal_surface=pygame.Surface((self.fractal_width,self.height))
        self.sidebar_surface=pygame.Surface((SIDEBAR_WIDTH,self.height))
        if TAICHI_AVAILABLE: self.pixels_ti=ti.Vector.field(3,dtype=ti.u8,shape=(self.fractal_width,self.height))
        slider_x=15; slider_w=SIDEBAR_WIDTH-110
        self.iter_slider=Slider(slider_x,200,slider_w,18,50,5000,self.max_iter,"Iter")
        self.power_slider=Slider(slider_x,240,slider_w,18,-10.0,10.0,self.power,"Power")
        self.supersample_slider=Slider(slider_x,280,slider_w,18,1,4,self.supersample,"SS")
        self.julia_re_slider=Slider(slider_x,320,slider_w,18,-2.0,2.0,self.julia_c_re,"Julia C_re")
        self.julia_im_slider=Slider(slider_x,360,slider_w,18,-2.0,2.0,self.julia_c_im,"Julia C_im")
        self.nova_R_slider=Slider(slider_x,320,slider_w,18,0.0,2.0,self.nova_R,"Nova R")
        self.nova_Cre_slider=Slider(slider_x,360,slider_w,18,-2.0,2.0,self.nova_C_re,"Nova C_re")
        self.nova_Cim_slider=Slider(slider_x,400,slider_w,18,-2.0,2.0,self.nova_C_im,"Nova C_im")
        self.needs_render=True; self.add_log(f"Window resized to {width}x{height}")

    def add_log(self,message): timestamp=time.strftime("%H:%M:%S"); self.log_messages.append(f"[{timestamp}] {message}")

    def handle_events(self):
        for event in pygame.event.get():
            if event.type==pygame.QUIT: pygame.quit(); sys.exit()
            sidebar_event=SimpleNamespace(type=event.type)
            if hasattr(event,'pos'): sidebar_event.pos=(event.pos[0]-self.fractal_width,event.pos[1])
            if hasattr(event,'button'): sidebar_event.button=event.button
            iter_ch=self.iter_slider.handle_event(sidebar_event); pow_ch=self.power_slider.handle_event(sidebar_event); ss_ch=self.supersample_slider.handle_event(sidebar_event)
            if iter_ch: self.auto_iter=False; self.max_iter=int(self.iter_slider.val); self.needs_render=True; self.add_log(f"Max iter set to {self.max_iter}")
            if pow_ch: self.power=float(self.power_slider.val); self.needs_render=True
            if ss_ch: self.supersample=int(round(self.supersample_slider.val)); self.needs_render=True
            if self.fractal_set_idx==1:
                if self.julia_re_slider.handle_event(sidebar_event): self.julia_c_re=float(self.julia_re_slider.val); self.needs_render=True; self.add_log(f"Julia C_re set to {self.julia_c_re:.3f}")
                if self.julia_im_slider.handle_event(sidebar_event): self.julia_c_im=float(self.julia_im_slider.val); self.needs_render=True; self.add_log(f"Julia C_im set to {self.julia_c_im:.3f}")
            if self.fractal_set_idx==6:
                if self.nova_R_slider.handle_event(sidebar_event): self.nova_R=float(self.nova_R_slider.val); self.needs_render=True
                if self.nova_Cre_slider.handle_event(sidebar_event): self.nova_C_re=float(self.nova_Cre_slider.val); self.needs_render=True
                if self.nova_Cim_slider.handle_event(sidebar_event): self.nova_C_im=float(self.nova_Cim_slider.val); self.needs_render=True
            if event.type==pygame.MOUSEBUTTONDOWN:
                if event.pos[0]<self.fractal_width: 
                    if event.button==1: self.recenter_from_pixel(*event.pos); self.needs_render=True; self.dragging=True; self.drag_start=event.pos; self.center_start=(self.center_x,self.center_y)
                    elif event.button==4: self.zoom_at(event.pos,0.8)
                    elif event.button==5: self.zoom_at(event.pos,1.25)
            if event.type==pygame.MOUSEBUTTONUP and event.button==1:
                if self.dragging: self.dragging=False; self.needs_render=True
            if event.type==pygame.MOUSEMOTION:
                if self.dragging and event.pos[0]<self.fractal_width:
                    dx=event.pos[0]-self.drag_start[0]; dy=event.pos[1]-self.drag_start[1]; aspect=float(self.fractal_width)/float(self.height); view_w=self.span*aspect; view_h=self.span
                    self.center_x=self.center_start[0]-dx/self.fractal_width*view_w; self.center_y=self.center_start[1]-dy/self.height*view_h; self.needs_render=True
            if event.type==pygame.KEYDOWN:
                if event.key==pygame.K_ESCAPE: pygame.quit(); sys.exit()
                if event.key==pygame.K_F11: self.fullscreen=not self.fullscreen; self.resize_window(*(self.monitor_size if self.fullscreen else self.default_window_size))
                if event.key==pygame.K_f: self.fractal_set_idx=(self.fractal_set_idx+1)%len(FRACTAL_SETS); self.needs_render=True; self.add_log(f"Switched to {FRACTAL_SETS[self.fractal_set_idx]}")
                if event.key==pygame.K_c: self.palette=(self.palette+1)%len(PALETTES); self.needs_render=True
                if event.key==pygame.K_r: self.reset_view(); self.needs_render=True
                if event.key==pygame.K_s: filename=f"fractal_{int(time.time())}.png"; pygame.image.save(self.fractal_surface,filename); self.add_log(f"Saved screenshot to {filename}")
                if event.key==pygame.K_a: self.auto_iter=not self.auto_iter; self.needs_render=True; self.add_log(f"Auto-iterations {'ON' if self.auto_iter else 'OFF'}")
                if event.key==pygame.K_p and MPMATH_AVAILABLE: self.mpmath_mode=not self.mpmath_mode; self.needs_render=True; self.add_log(f"MPMATH mode {'ON' if self.mpmath_mode else 'OFF'}")
                if event.key==pygame.K_LEFTBRACKET: self.mpmath_precision=max(15,self.mpmath_precision-10); self.needs_render=True; self.add_log(f"MPMATH dps set to {self.mpmath_precision}")
                if event.key==pygame.K_RIGHTBRACKET: self.mpmath_precision=self.mpmath_precision+10; self.needs_render=True; self.add_log(f"MPMATH dps set to {self.mpmath_precision}")

    def zoom_at(self,pos,zoom_factor):
        mx,my=pos; aspect=float(self.fractal_width)/float(self.height); view_w=self.span*aspect; view_h=self.span
        wx=self.center_x-view_w/2.0+(mx/self.fractal_width)*view_w; wy=self.center_y-view_h/2.0+(my/self.height)*view_h
        self.span*=zoom_factor; self.center_x=wx+(self.center_x-wx)*zoom_factor; self.center_y=wy+(self.center_y-wy)*zoom_factor; self.needs_render=True

    def recenter_from_pixel(self,mx,my):
        aspect=float(self.fractal_width)/float(self.height); view_w=self.span*aspect; view_h=self.span
        self.center_x=self.center_x-view_w/2.0+(mx/self.fractal_width)*view_w; self.center_y=self.center_y-view_h/2.0+(my/self.height)*view_h

    def reset_view(self):
        if self.fractal_set_idx in [5,6]: self.center_x, self.center_y = 0.0, 0.0
        elif self.fractal_set_idx == 2: self.center_x, self.center_y = -0.5, -1.0
        else: self.center_x, self.center_y = -0.5, 0.0
        self.span=3.5; self.power=2.0; self.max_iter=200; self.iter_slider.val=self.max_iter; self.power_slider.val=self.power; self.auto_iter=True
        self.add_log("View reset to defaults.")

    def render(self):
        current_iter = self.max_iter
        if self.dragging and self.auto_iter: current_iter = max(50, int(self.max_iter * 0.5))
        if not self.dragging: self.add_log(f"Rendering {FRACTAL_SETS[self.fractal_set_idx]}...")
        start=time.time()
        if self.auto_iter and not self.dragging: self.max_iter=int(max(150,70*abs(math.log10(max(1e-300,self.span))))); self.iter_slider.val=self.max_iter
        use_gpu=TAICHI_AVAILABLE and not self.mpmath_mode
        if use_gpu:
            target_dtype=np.float32 if GPU_PRECISION==ti.f32 else np.float64
            fractal_kernel_ti(self.pixels_ti,self.fractal_width,self.height,self.fractal_set_idx,target_dtype(self.power),int(current_iter),
                target_dtype(self.julia_c_re),target_dtype(self.julia_c_im),target_dtype(self.center_x),
                target_dtype(self.center_y),target_dtype(self.span),target_dtype(self.escape_radius),int(self.palette),
                target_dtype(self.nova_R), target_dtype(self.nova_C_re), target_dtype(self.nova_C_im))
            img = self.pixels_ti.to_numpy()
        else:
            img = render_fractal_numpy(self.fractal_width,self.height,self.center_x,self.center_y,self.span,int(current_iter),self.power,self.julia_c_re,self.julia_c_im,
                fractal_set_idx=self.fractal_set_idx,escape_radius=self.escape_radius,supersample=self.supersample,palette=self.palette,
                use_mpmath=self.mpmath_mode,precision=self.mpmath_precision, nova_R=self.nova_R, nova_C=(self.nova_C_re, self.nova_C_im))
        pygame.surfarray.blit_array(self.fractal_surface, img)
        self.last_render_time=time.time()-start; self.needs_render=False
        if not self.dragging: self.add_log(f"Render finished in {self.last_render_time:.3f}s")

    def draw_sidebar(self):
        self.sidebar_surface.fill((25,28,32)); title_txt=self.font_big.render("Fractal Explorer",True,(220,220,220)); self.sidebar_surface.blit(title_txt,(15,15))
        renderer_name="GPU" if TAICHI_AVAILABLE and not self.mpmath_mode else "CPU"
        if self.mpmath_mode: renderer_name+=f" (MPMATH {self.mpmath_precision} dps)"
        elif TAICHI_AVAILABLE: renderer_name+=f" ({'f32' if GPU_PRECISION==ti.f32 else 'f64'})"
        lines=[f"Set: {FRACTAL_SETS[self.fractal_set_idx]} (F)",f"Center X: {self.center_x:.6f}",f"Center Y: {self.center_y:.6f}",f"Span: {self.span:.2e}",f"Palette: {PALETTES[self.palette]} (C)",f"Renderer: {renderer_name} (P)"]
        for i,ln in enumerate(lines): self.sidebar_surface.blit(self.font.render(ln,True,(200,200,200)),(15,60+i*22))
        self.iter_slider.draw(self.sidebar_surface,self.font); self.supersample_slider.draw(self.sidebar_surface,self.font)
        if self.fractal_set_idx in [0, 2, 3, 4, 1]: self.power_slider.draw(self.sidebar_surface, self.font)
        if self.fractal_set_idx == 1: self.julia_re_slider.draw(self.sidebar_surface, self.font); self.julia_im_slider.draw(self.sidebar_surface, self.font)
        if self.fractal_set_idx == 6: self.nova_R_slider.draw(self.sidebar_surface, self.font); self.nova_Cre_slider.draw(self.sidebar_surface, self.font); self.nova_Cim_slider.draw(self.sidebar_surface, self.font)
        log_y_start = self.height - 250
        pygame.draw.rect(self.sidebar_surface,(40,45,50),(10,log_y_start,SIDEBAR_WIDTH-20,240)); pygame.draw.rect(self.sidebar_surface,(80,85,90),(10,log_y_start,SIDEBAR_WIDTH-20,240),1)
        log_title=self.font.render("Event Log",True,(200,200,200)); self.sidebar_surface.blit(log_title,(20,log_y_start+5))
        for i,msg in enumerate(reversed(self.log_messages)):
            if (self.height-30-i*18) < log_y_start + 25: break
            log_text=self.font.render(msg,True,(170,170,170)); self.sidebar_surface.blit(log_text,(20,self.height-30-i*18))

    def run(self):
        while True:
            self.handle_events()
            render_now = self.needs_render
            if self.dragging:
                now = time.time()
                if now - self.last_pan_render_time < (1.0 / PAN_RENDER_FPS): render_now = False
                else: self.last_pan_render_time = now
            if render_now:
                if self.mpmath_mode:
                    feedback_surf=self.fractal_surface.copy(); box=pygame.Surface((300,50),pygame.SRCALPHA); box.fill((0,0,0,180)); feedback_surf.blit(box,(self.fractal_width//2-150,self.height//2-25))
                    msg=self.font.render(f"Rendering (mpmath {self.mpmath_precision} dps)...",True,(255,255,100)); feedback_surf.blit(msg,(self.fractal_width//2-msg.get_width()//2,self.height//2-msg.get_height()//2))
                    self.screen.blit(feedback_surf,(0,0)); pygame.display.flip()
                self.render()
            self.screen.blit(self.fractal_surface,(0,0)); self.draw_sidebar(); self.screen.blit(self.sidebar_surface,(self.fractal_width,0))
            pygame.display.flip(); self.clock.tick(FPS_LIMIT)

if __name__ == '__main__':
    app = FractalApp()
    app.run()
