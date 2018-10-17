#!/usr/bin/env python
# coding: utf-8


# The MIT License (MIT)

# Copyright (c) 2017 Iurii Zykov

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
# IN THE SOFTWARE.


import ctypes
import math
import sys
import struct


# from <variant>.h
# accurate function we are approximating
# (if known, only for accurate solution printing)
def p(x, y):
  return 2 / (1 + x * x + y * y)


def abs_error(x, y):
  return x - y

def rel_error(x, y):
  return abs_error(x, y) / y if y != 0 else x


def dump_plt_file(plt_filename, data_filename):
  width = 1280
  height = 960
  with open(plt_filename, 'w') as plt_file:
    plt_file.write('#!/usr/bin/env gnuplot')
    plt_file.write('set terminal png size %s, %s\n' % (width, height))
    plt_file.write('set output "%s.png"\n' % data_filename)
    plt_file.write('set xlabel "x"\n')
    plt_file.write('set ylabel "y"\n')
    plt_file.write('set zlabel "p(x, y)"\n')
    plt_file.write('set ticslevel 0\n')
    plt_file.write('set pal defined\n')
    plt_file.write('set pm3d implicit at s\n')
    plt_file.write('set style line 1 lt 1 lw 1 pt -1\n')
    plt_file.write('splot "%s" ls 1 pal\n' % data_filename);


if __name__ == '__main__':
  if len(sys.argv) < 2:
    print('Usage: %s <input>' % sys.argv[0])
    sys.exit(0)
  data = open(sys.argv[1], 'rb').read()
  print('Size: %s bytes' % len(data))
  order = '>' if len(sys.argv) >= 3 and sys.argv[2] == 'be' else '<'
  dtype = struct.unpack('%sc' % order, data[:1])
  data = data[1:]
  dtype = 'd' if dtype else 'f'
  ctype = ctypes.c_double if dtype == 'd' else ctypes.c_float
  print('Datatype: %s (%s bytes)' % (ctype, ctypes.sizeof(ctype)))
  N1, N2 = struct.unpack('%sii' % order, data[:2 * ctypes.sizeof(ctypes.c_int)])
  print('N1 = %s\nN2 = %s' % (N1, N2))
  data = data[2 * ctypes.sizeof(ctypes.c_int):]
  grid = {'X': [], 'Y': []}
  for i in range(N1 + 1):
    grid['X'].append(struct.unpack('%s%s' % (order, dtype),
      data[i * ctypes.sizeof(ctype):(i + 1) * ctypes.sizeof(ctype)])[0])
  data = data[(N1 + 1) * ctypes.sizeof(ctype):]
  for i in range(N2 + 1):
    grid['Y'].append(struct.unpack('%s%s' % (order, dtype),
      data[i * ctypes.sizeof(ctype):(i + 1) * ctypes.sizeof(ctype)])[0])
  data = data[(N2 + 1) * ctypes.sizeof(ctype):]
  f = []
  for i in range(N1 + 1):
    f.append([])
    for j in range(N2 + 1):
      f[i].append(struct.unpack('%s%s' % (order, dtype),
        data[(i * (N2 + 1) + j) * ctypes.sizeof(ctype)
        :(i * (N2 + 1) + j + 1) * ctypes.sizeof(ctype)])[0])
  data = data[(N1 + 1) * (N2 + 1) * ctypes.sizeof(ctype):]
  
  try:
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import mpl_toolkits.mplot3d.axes3d as axes3d
  except ImportError:
    # numpy is already required by mpl
    print('no matplotlib, skipping...')
    pass
  else:
    f = np.array(f)
    X, Y = np.meshgrid(grid['X'], grid['Y'], indexing='ij')
    fig = plt.figure(num='Approximation')
    ax = axes3d.Axes3D(fig)
    ax.scatter(X, Y, f, color='red', alpha=0.25, s=1)
    ax.plot_wireframe(X, Y, f, color='blue', alpha=0.25)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.savefig('%s.png' % sys.argv[1])
    plt.clf()
    f = []
    for i in range(N1 + 1):
      f.append([])
      for j in range(N2 + 1):
        f[i].append(p(grid['X'][i], grid['Y'][j]))
    f = np.array(f)
    fig = plt.figure(num='Accurate')
    ax = axes3d.Axes3D(fig)
    ax.scatter(X, Y, f, color='red', alpha=0.25, s=1)
    ax.plot_wireframe(X, Y, f, color='blue', alpha=0.25)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.savefig('%s.accur.png' % sys.argv[1])
    plt.clf()
  
  with open(sys.argv[1] + '.accur.dat', 'w') as data_file:
    data_file.write('# x y p\n')
    for i in range(N1 + 1):
      for j in range(N2 + 1):
        data_file.write('%s %s %s\n' % (
          grid['X'][i],
          grid['Y'][j],
          p(grid['X'][i], grid['Y'][j])
        ))
      data_file.write('\n')
  dump_plt_file(sys.argv[1] + '.accur.plt', sys.argv[1] + '.accur.dat')
  
  with open(sys.argv[1] + '.dat', 'w') as data_file:
    data_file.write('# x y p\n')
    for i in range(N1 + 1):
      for j in range(N2 + 1):
        data_file.write('%s %s %s\n' % (
          grid['X'][i],
          grid['Y'][j],
          f[i][j]
        ))
      data_file.write('\n')
  dump_plt_file(sys.argv[1] + '.plt', sys.argv[1] + '.dat')
  
  max_abs_error = 0
  with open(sys.argv[1] + '.ae.dat', 'w') as data_file:
    data_file.write('# x y p\n')
    for i in range(N1 + 1):
      for j in range(N2 + 1):
        err = abs_error(f[i][j], p(grid['X'][i], grid['Y'][j]))
        data_file.write('%s %s %s\n' % (
          grid['X'][i],
          grid['Y'][j],
          err
        ))
        max_abs_error = max(max_abs_error, abs(err))
      data_file.write('\n')
  dump_plt_file(sys.argv[1] + '.ae.plt', sys.argv[1] + '.ae.dat')
  
  max_rel_error = 0
  with open(sys.argv[1] + '.re.dat', 'w') as data_file:
    data_file.write('# x y p\n')
    for i in range(N1 + 1):
      for j in range(N2 + 1):
        err = rel_error(f[i][j], p(grid['X'][i], grid['Y'][j]))
        data_file.write('%s %s %s\n' % (
          grid['X'][i],
          grid['Y'][j],
          err
        ))
        max_rel_error = max(max_rel_error, abs(err))
      data_file.write('\n')
  dump_plt_file(sys.argv[1] + '.re.plt', sys.argv[1] + '.re.dat')
  print('max absolute error: %s' % max_abs_error)
  print('max relative error: %s' % max_rel_error)
