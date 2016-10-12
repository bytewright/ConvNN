conv_filter = [5, 4]
conv_stride = [2, 2]
#conv_filter = [11]
#conv_stride = [4]
size = 227
output_vars = 0
for i in range(len(conv_filter)):
    print('conv_filter:{}\nconv_stride:{}\nsize:{}'.format(conv_filter[i], conv_stride[i], size))
    start = 0
    end = start + conv_filter[i]
    output_vars = 1
    while end < size:
        print('{}\t({}, {})'.format(output_vars, start, end))
        if end + conv_stride[i] > size:
            break
        start += conv_stride[i]
        end += conv_stride[i]
        output_vars+=1
    if i < len(conv_filter)-1:
        size = output_vars
        output_vars = 1
        start = 0

print('endsize:\t{}x{}'.format(output_vars, output_vars))
