'''
Autor: Adrián Robles Arques

Código para simulación de la difusión del calor
Usando la biblioteca Plotly
'''
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
import plotly.graph_objects as go

def frame_args(duration):
    return {
            "frame": {"duration": duration},
            "mode": "immediate",
            "fromcurrent": True,
            "transition": {"duration": duration, "easing": "linear"},
            }
            

# ===================================
#        Un material 3D (FTCS) - Cubo de cobre
# ===================================

Lx = 1. #Longitud del lado de la placa en m
Ly = 1.
Lz = 1.
Nx = 30
Ny = 30
Nz = 30
dx = Lx/Nx #paso espacial
dy = Ly/Ny
dz = Lz/Nz
NT = 500 #número de pasos temporales
D = 1.13e-4 #Cobre
#D = 2.02e-5 #Difusividad aire
#paso temporal que cumple la condicion de Courant
dt = 0.1*(dx*dy*dz)**2/(2*D*((dy*dx)**2+(dy*dz)**2+(dx*dz)**2))
Tlow = 300.0
Thigh = 700.0

pasos = 20

t = np.linspace(0,dt*NT,NT+1)
u = np.zeros((NT+1,Nx+1,Ny+1,Nz+1),float)

u[:,0,:,:] = Tlow
u[:,:,0,:] = Tlow
u[:,:,:,0] = Tlow
u[:,Nx,:,:] = Tlow
u[:,:,Ny,:] = Tlow
u[:,:,:,Nz] = Tlow
for i in range(1,Ny):
    for j in range(1,Nx):
      for m in range(1,Nz):
        if np.sqrt((i-Ny/2)**2+(j-Nx/2)**2+(m-Nz/2)**2) >= 10:
            u[0,i,j,m] = Tlow
        else:
            u[0,i,j,m] = Thigh

cont = 0

for k in range(0,NT):
    cont += 1
    u[k+1,1:-1,1:-1,1:-1] = u[k,1:-1,1:-1,1:-1] + D*dt*((u[k,2:,1:-1,1:-1]+
     u[k,:-2,1:-1,1:-1]-2*u[k,1:-1,1:-1,1:-1])/dx**2 + (u[k,1:-1,2:,1:-1] + 
      u[k,1:-1,:-2,1:-1]-2*u[k,1:-1,1:-1,1:-1])/dy**2+(u[k,1:-1,1:-1,2:] + 
       u[k,1:-1,1:-1,:-2]-2*u[k,1:-1,1:-1,1:-1])/dz**2)
    print(str(round(cont*100/NT,3)) + '%')

#-----------------------------------------------------------------------------
# ANIMACION 3D CON PLOTLY
#-----------------------------------------------------------------------------

X, Y, Z = np.mgrid[-Lx/2:Lx/2:31j, -Ly/2:Ly/2:31j, -Lz/2:Lz/2:31j]
values = u[0]

fig = go.Figure(data=[go.Volume(
    x=X.flatten(),
    y=Y.flatten(),
    z=Z.flatten(),
    value=values.flatten(),
    isomin= Tlow,
    isomax= Thigh,
    opacity=0.1, # needs to be small to see through all surfaces
    surface_count=21, # needs to be a large number for good volume rendering,
    )])
    
names = [('t = {0}'.format(round(i*dt),4)) for i in range(0,NT+1,pasos)] 
#Creamos los frames de la animación

frames = []
count = 0
for i in range(0,NT+1,pasos):
    frames.append(go.Frame(data = [go.Volume(x=X.flatten(),
        y=Y.flatten(),
        z=Z.flatten(),
        value = u[i].flatten(),
        text = names[count] + ' seg.',
        name = names[count],
        isomin= Tlow,
        isomax= Thigh,
        opacity = 0.1,
        surface_count = 31,)]))
    count += 1

fig.frames = frames

#Creamos el slider
slider = [dict(steps = [dict(method = 'animate',
                              args= [f,                           
                              dict(mode= 'immediate',
                                   frame = dict(duration = 100, redraw=True),
                                   transition=dict(duration= 0))
                                 ],
                              label = names[k] + ' seg')
                              for k, f in enumerate(fig.frames)],
                active=0,
                transition= dict(duration = 0 ),
                x=0, # slider starting position  
                y=0, 
                currentvalue=dict(font=dict(size=12), 
                                  prefix='frame: ', 
                                  visible=True, 
                                  xanchor= 'center'
                                 ),  
                len=1.0) #slider length
           ]


botones = [
            {
                "buttons": [
                    {
                        "args": [None, frame_args(20)],
                        "label": "&#9654;", # play symbol
                        "method": "animate",
                    },
                    {
                        "args": [[None], frame_args(0)],
                        "label": "&#9208;", # pause symbol
                        "method": "animate",
                    },
                ],
                "direction": "left",
                "pad": {"r": 20, "t": 70},
                "type": "buttons",
                "x": 0.1,
                "y": 1,
            }
         ]


fig.update_layout(
        title_text = "Difusión de temperatura", hovermode = "closest",
        updatemenus = botones,
        sliders = slider)
                   
fig.show()


# ===================================
#     Varios Materiales 3D (FTCS) - Habitación con suelo térmico - 
# ===================================

Lx = 5. #Longitud del lado de la habitación en m
Ly = 5.
Lz = 3.
Nx = 35
Ny = 35
Nz = 20
dx = Lx/Nx #paso espacial
dy = Ly/Ny
dz = Lz/Nz
NT = 200 #número de pasos temporales
"Habitación Vacía"
D1 = 1.16e-6 #Difusividad piedra
D2 = 2.02e-5 #Difusividad aire
D3 = 4.91e-7 #Difusividad ladrillo arcilla
D4 = 5.61e-7 #Difusividad vidrio plano

pasos = 5

#paso temporal que cumple la condicion de Courant
dt = 0.9*(dx*dy*dz)**2/(2*max(D1,D2,D3,D4)*((dx*dy)**2+(dy*dz)**2+(dx*dz)**2))

D = np.zeros((Nx+1,Ny+1,Nz+1),float)
D[:,:,:] = D2 #Aire de la habitación
D[:,:,0:1] = D1 #Suelo térmico de piedra
D[0:1,:,:] = D3 #Paredes de ladrillo
D[Nx-1:Nx,:,:] = D3
D[:,0:1,:] = D3
D[:,Ny-1:Ny,:] = D3
D[:,:,Nz-1:Nz] = D3
D[0:1,12:22,8:13] = D4 #Ventanas
D[12:22,Ny-1:Ny,8:13] = D4


Tlow = 278.
Thigh = 310.

t = np.linspace(0,dt*NT,NT+1)
u = np.zeros((NT+1,Nx+1,Ny+1,Nz+1),float)

u[:,:,:,:] = Tlow + 5. #Interior de la habitación
u[:,:,:,0:1] = Thigh #Suelo térmico de piedra
u[:,0,:,:] = Tlow + 5. #Paredes de ladrillo
u[0,0:1,12:22,8:13] = Tlow-5 #Ventanas
u[0,12:22,0:1,8:13] = Tlow-5


cont = 0

for k in range(0,NT):
    cont += 1
    u[k+1,1:-1,1:-1,1:-1] = u[k,1:-1,1:-1,1:-1] + D[1:-1,1:-1,1:-1]*dt*(
    (u[k,2:,1:-1,1:-1]+u[k,:-2,1:-1,1:-1]-2*u[k,1:-1,1:-1,1:-1])/dx**2 + 
    (u[k,1:-1,2:,1:-1] + u[k,1:-1,:-2,1:-1]-2*u[k,1:-1,1:-1,1:-1])/dy**2+
    (u[k,1:-1,1:-1,2:] + u[k,1:-1,1:-1,:-2]-2*u[k,1:-1,1:-1,1:-1])/dz**2)
    u[k+1,:,:,0:1] = Thigh #Suelo térmico de piedra
    u[k+1,0:1,12:22,8:13] = Tlow-5 #Ventanas
    u[k+1,12:22,0:1,8:13] = Tlow-5
    
    print(str(round(cont*100/NT,3)) + '%')

#-----------------------------------------------------------------------------
# ANIMACION 3D CON PLOTLY
#-----------------------------------------------------------------------------

X, Y, Z = np.mgrid[-Lx/2:Lx/2:36j, -Ly/2:Ly/2:36j, -Lz/2:Lz/2:21j]
values = u[0]

#Creamos la figura inicial
fig = go.Figure(data=[go.Volume(
    x=X.flatten(),
    y=Y.flatten(),
    z=Z.flatten(),
    value=values.flatten(),
    isomin= Tlow,
    isomax= Thigh,
    opacity=0.1, # needs to be small to see through all surfaces
    surface_count=21, # needs to be a large number for good volume rendering,
    )])  

names = [('t = {0}'.format(round(i*dt/60),2)) for i in range(0,NT+1,pasos)] 
#Creamos los frames de la animación
frames = []
count = 0
for i in range(0,NT+1,pasos):
    frames.append(go.Frame(data = [go.Volume(x=X.flatten(),
        y=Y.flatten(),
        z=Z.flatten(),
        value = u[i].flatten(),
        name = names[count],
        text = names[count] + ' min.',
        isomin= Tlow-5,
        isomax= Thigh,
        opacity = 0.1,
        surface_count = 12,)]))
    count += 1

fig.frames = frames

#Creamos el slider
slider = [dict(steps = [dict(method = 'animate',
                              args= [f,                           
                              dict(mode= 'immediate',
                                   frame= dict(duration=100, redraw=True),
                                   transition=dict(duration= 0))
                                 ],
                              label = names[k] + ' min')
                              for k, f in enumerate(fig.frames)],
                active=0,
                transition= dict(duration= 0 ),
                x=0, # slider starting position  
                y=0, 
                currentvalue = dict(font=dict(size=12), 
                                  prefix='frame: ', 
                                  visible = True, 
                                  xanchor= 'center'
                                 ),  
                len=1.0) #slider length
           ]


botones = [
            {
                "buttons": [
                    {
                        "args": [frames, frame_args(20)],
                        "label": "&#9654;", # play symbol
                        "method": "animate",
                    },
                    {
                        "args": [[None], frame_args(0)],
                        "label": "&#9208;", # pause symbol
                        "method": "animate",
                    },
                ],
                "direction": "left",
                "pad": {"r": 20, "t": 70},
                "type": "buttons",
                "x": 0.1,
                "y": 1,
            }
         ]


fig.update_layout(
        title_text = "Difusión de temperatura", hovermode = "closest",
        updatemenus = botones,
        sliders = slider)
                   
fig.show()