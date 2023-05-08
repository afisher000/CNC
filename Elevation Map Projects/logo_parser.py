# %%


# Read lines from gcode
logo_file = 'logo_fusion.nc'
with open(logo_file, 'r') as f:
    lines = f.readlines()

# Initialize position
position = {'X':0, 'Y':0, 'Z':0}

# Create list for holding paths
paths, path = [], []

# Separate line into commands
for line in lines:
    commands = line.strip('\n').split(' ')
    for command in commands:
        if len(command)>0:
            # Check if comment
            if command[0]=='(':
                continue
            # If X,Y, or Z command, update position
            elif command[0] in position.keys():
                position[command[0]] = float(command[1:])
                print(f"X={position['X']:.2f}, Y={position['Y']:.2f}, Z={position['Z']:.2f}")

        # If z is cutting, add to path
        if position['Z']<1:
            path.append([position['X'], position['Y'], position['Z']])
        else:
            # Add completed path to paths, reinitialize path
            paths.append(path)
            path = []

