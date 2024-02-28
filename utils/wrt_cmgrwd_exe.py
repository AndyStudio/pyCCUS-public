import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--sim_sr3', default='CCS_GEM')
parser.add_argument('--ext_rwo', default=None)
parser.add_argument('--proplist', default=['SG','PRES'], nargs='+', type=str)
parser.add_argument('--layer_num', default=1, nargs='+', type=int)
parser.add_argument('--time_step', default='ALL-TIMES')
parser.add_argument('--precis', default=4, type=int)
args = parser.parse_args()

def wrt_cmgrwd():
    print(args.sim_sr3)
    print(args.ext_rwo)
    print(args.proplist)
    print(args.layer_num)
    
    line1 = '*FILES    ' + args.sim_sr3 + '.sr3'
    line2 = '*PRECISION '+ str(args.precis)
    
    if args.ext_rwo is None:
        ext_file_name = args.sim_sr3
    else:
        ext_file_name = args.ext_rwo
    
    for prop in args.proplist:
        for layer in args.layer_num:
            extline1 = '*OUTPUT    ' + ext_file_name + '.rwo'
            extline2 = '*PROPERTY-FOR' + prop + args.time_step + 'XYZLAYER' + str(layer)
            
            print(extline1)
            print(extline2)
    
#    print('FILES    ' + args.sim_sr3 + '.sr3')
#    print('PRECISION '+ str(args.precis))

    
    
if __name__ == '__main__':
    wrt_cmgrwd()
