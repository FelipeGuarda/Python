import subprocess, os

target = r'C:\Users\USUARIO\SynologyDrive\2. Camaras trampa (SC)\SynologyDrive\DATOS_GRILLA CÁMARAS TRAMPA\2. CAMPAÑAS DE RECOLECCION DE IMAGENES\Primavera 2025'
link   = r'C:\ADDAX\Primavera_2025'

print(f'Target exists: {os.path.isdir(target)}')
print(f'Link path:     {link}')
print()

if os.path.exists(link):
    print('Junction already exists.')
else:
    cmd = f'mklink /J "{link}" "{target}"'
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True, encoding='oem')
    print('stdout:', result.stdout)
    print('stderr:', result.stderr)
    print('Return code:', result.returncode)

print()
print('Verifying...')
print(f'Junction accessible: {os.path.isdir(link)}')
print(f'File count: {sum(len(f) for _, _, f in os.walk(link))}')
