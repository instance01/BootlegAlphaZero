import pstats

name = '2.profile'
out = pstats.Stats(name)

print('sort by cumulative')
out.sort_stats('cumulative').print_stats(40)

# print('sort by total time')
# out.sort_stats('time').print_stats(20)
