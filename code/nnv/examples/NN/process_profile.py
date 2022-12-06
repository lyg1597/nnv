import pstats 
p = pstats.Stats('res.txt') 
p.sort_stats('cumtime').print_stats()