conv_filter = [7, 3, 3]
conv_stride = [3, 1, 2]
padding =     [3, 0, 0]
#conv_filter = [11]
#conv_stride = [4]
size = 227
output_vars = 0
for F,P,S in zip(conv_filter,padding,conv_stride):
    out_dim = (float(size)-float(F)+2.0*float(P))/float(S)+1.0
    print 'F {},P {},S {}=\t{}x{}'.format(F, P, S, int(out_dim),int(out_dim))
    size = out_dim
print 'outsize: {}x{}'.format(int(out_dim),int(out_dim))

