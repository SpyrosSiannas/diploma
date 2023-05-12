import os
import subprocess

rootdir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
tmc3_dir = os.path.join(rootdir, 'submodules/mpeg-pcc-tmc13/build/tmc3')

def validate_tmc3():
    if not os.path.exists(os.path.join(tmc3_dir, 'tmc3')):
        raise Exception('tmc3 not found, please build it first via build.bash inside the root dir.')
    else:
        print("TMC3 found. Continuing...")

def gpcc_encode(filedir, bin_dir, show=False):
    """Compress point cloud losslessly using MPEG G-PCCv14. 
    

    Args:
    ----
    filedir: str the input point cloud file path
    bin_dir: str the output encoded file path
    show: bool whether to show the output of the subprocess.

    Raises:
    ------
    Exception: prompts user to build tmc3 if it is not found.
    """
    validate_tmc3()
    subp=subprocess.Popen(rootdir+'/tmc3'+ 
                            ' --mode=0' + 
                            ' --positionQuantizationScale=1' + 
                            ' --trisoupNodeSizeLog2=0' + 
                            ' --neighbourAvailBoundaryLog2=8' + 
                            ' --intra_pred_max_node_size_log2=6' + 
                            ' --inferredDirectCodingMode=0' + 
                            ' --maxNumQtBtBeforeOt=4' +
                            ' --uncompressedDataPath='+filedir + 
                            ' --compressedStreamPath='+bin_dir, 
                            shell=True, stdout=subprocess.PIPE)
    c=subp.stdout.readline()
    while c:
        if show: 
            print(c)
        c=subp.stdout.readline()
    
    return 

def gpcc_decode(bin_dir, rec_dir, show=False):
    subp=subprocess.Popen(rootdir+'/tmc3'+ 
                            ' --mode=1'+ 
                            ' --compressedStreamPath='+bin_dir+ 
                            ' --reconstructedDataPath='+rec_dir+
                            ' --outputBinaryPly=0'
                          ,
                            shell=True, stdout=subprocess.PIPE)
    c=subp.stdout.readline()
    while c:
        if show: 
            print(c)      
        c=subp.stdout.readline()
    
    return

if __name__ == "__main__":
    validate_tmc3()