import numpy as np
import os


class gcode_manager():
    def __init__(self, feed_rate=2000, laser_power=250, zsafe=5, xsafe=-20):
        ''' Parameters
        feed_rate:      speed of motors (mm/min)
        laser_power:    laser power from 0 to 250
        zsafe:          Safe zheight for bit. 
        xsafe:          Safe xposition for bit to drop to lower height for roughing
        time:           Estimate runtime
        gcode:          Gcode commands
        '''

        # Initialize parameters
        self.feed_rate      = feed_rate
        self.laser_power    = laser_power
        self.zsafe          = zsafe
        self.xsafe          = xsafe
        self.time           = 0
        self.gcode          = f'G90 G94\nG17\nG21\nG90\nG54\nF{feed_rate}\nM3 S0\n\n\n'

        # Checks
        if laser_power<0 or laser_power>250:
            raise ValueError('Laser power must be value between 0 and 250')
        
        if feed_rate<100 or feed_rate>2500:
            raise ValueError('Feed rate must be between 100 and 2500')
        
        return
    

    def laser_on(self):
        ''' Turn laser on '''
        self.gcode += f'S{self.laser_power}\n'
        return
    

    def laser_off(self):
        ''' Turn laser off '''
        self.gcode += 'S0\n'
        return
    

    def save(self, file):
        ''' Save gcode to file. '''

        # Lift and send to origin
        self.lift(0)
        self.gcode += f'G0 X0 Y0'

        # Make directory if it does not exist
        directory = os.path.dirname(file)
        if not os.path.exists(directory):
            os.makedirs(directory)

        
        with open(file, 'w') as f:
            f.write(self.gcode)

        print(f'\tSaved "{file}"\n\tEstimated runtime = {self.time:.1f} min\n')
        return
    

    def lift(self, z=None):
        ''' Change zheight. Raise to zsafe if no input. '''
        self.gcode += f'G0 Z{self.zsafe:.2f}\n' if z is None else f'G0 Z{z:.2f}\n'
        return
    

    def add_path(self, xs, ys, zs, laser=False, roughing=False, dasharray='none'):
        # Even if not using laser, might as well turn it on to simplify code?

        ''' Add commands to move along specified path'''
        # Move to first position
        if roughing:
            self.gcode += f'G0 X{self.xsafe:.2f} Y0\n'
            self.gcode += f'G0 Z{zs[0]:.2f}\n'
        else:
            self.gcode += f'G0 X{xs[0]:.2f} Y{ys[0]:.2f} Z{zs[0]:.2f}\n'

        # Turn on laser
        if laser:
            self.laser_on()

        # Move along points in path
        if dasharray is None or laser==False: #no dashed line
            lines = [f'G1 X{x:.2f} Y{y:.2f} Z{z:.2f}\n' for x,y,z in
                    zip(xs, ys, zs)]
        else: #dashed line
            lines = []
            js_turnoff = np.cumsum(dasharray)[0:len(dasharray):2]
            js_turnon = np.cumsum(dasharray)[1:len(dasharray):2]
            for j, (x,y,z) in enumerate(zip(xs, ys, zs)):
                jmod = 1+np.mod(j-1, sum(dasharray)) #between 1 and sum(dasharray) inclusive
                if jmod in js_turnon:
                    lines.append(f'S{self.laser_power}\n')
                elif jmod in js_turnoff:
                    lines.append(f'S0\n')
                lines.append(f'G1 X{x:.2f} Y{y:.2f} Z{z:.2f}\n')
        self.gcode += ''.join(lines)

        # Turn off laser
        if laser:
            self.laser_off()
        
        # Estimate time
        distances = np.sqrt(np.diff(xs)**2+np.diff(ys)**2+np.diff(zs)**2)
        self.time += distances.sum()/self.feed_rate

        return 


