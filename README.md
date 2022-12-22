# Erawan Cluster


## โครงสร้างระบบโดยรวม

### Hardware Spec 
1 Login node

3 Compute node with 8 GPU Nvidia A100 (CPU 128 Cores, RAM 2 TB)

### Storage

-   Archive 288TB (Usable)
-   Parallel File System (wekaio) 150TB

### Network

-   25Gbps
-   200Gbps Infiniband

### Nework share directory (via NFS)

- /home 288TB (Read, Write)
- /opt/ohpc/pub (Read only)


#### NCBI blast database

ตอนนี้ /home/sharedata/blast/db หลังจากระบบเรียบร้อยแล้วจะเปลี่ยนเป็น /sharedata/blast/db โดยตอนนี้มีข้อมูลต่อไปนี้วางอยู่แล้ว
-   nr
-   nt
-   refseq_protein
-   refseq_rna
-   swissprot

Compute node Local disk

/scratch.local

      
### Software
-   OS: Rocky linux 8.7
-   OpenHPC 2.4


### List of application software
| Application software | Versions |
|--|--|
| 2.1 Python | 3.6.8 (default), 3.8.13, 3.9.13, 2.7.18 |
| 2.2 TensorFlow | 2.6.2 |
| 2.3 Anaconda | 3-2022.05 |
| 2.4 Keras | 2.6.0 |
| 2.5 Pytorch | 1.10.2+cu113 |
| 2.6 openCV | 3.4.6 |
| 2.7 R program | 4.2.2 |
| 2.8 Transformers | 4.18.0 |
| 2.9 AMPL | 20221023 |
| 2.10 C language | 9.4.0 , 8.5.0 |
| 2.11 Clara Train SDK | 4.1 |
| 2.12 CUDA Toolkit | 11.1, 11.8, 12.0 |
| 2.13 CuDNN | 8.7 |
| 2.14 GCC | 9.4.0 , 8.5.0 |
| 2.15 GNU C++ | 9.4.0 , 8.5.0 |
| 2.16 Matplotlib | 3.0.3 |
| 2.17 NumPy | 1.19.5 , 1.14.3 |
| 2.18 Open MPI | 4.1.1 (gcc) , 4.1.4 (intel) |
| 2.19 pandas | 0.25.3 |
| 2.20 PGI Compiler (NVHPC-2022) | 22.11 |
| 2.21 Ray | 2.1.0 |
| 2.22 Julia | 1.8.3 |
| 2.23 Mkl (bundle with Intel One API) | 2022.2.1 |
| 3.1 Jupyter notebook | 1.13.5 |
| 3.2 Gurobi | 10.0 |
| 3.3 GROMACS | 2022.4 |
| 3.4 BLAST | 2.13.0 |
| 3.5 LAMMPS | 202105027 |
| 3.6 LINGO | 19 |
| 3.7 Quantum Espresso | 6.8 |
| 3.8 Singularity | 3.7.1-5.1.ohpc.2.1 |
| 3.9 ABINIT | 9.6.2 |
| 3.10 CP2K | 9.1.0 |
| 3.11 DL_POLY | 4 |
| 3.12 FreeSurfer | 7.3.2 |
| 3.13 NAMD | 2.14 |
| 3.14 NWChem | 7.0.2 |
| 3.15 OpenFOAM | 10 |
| 3.16 ORCA | 5.0.3 |
| 3.17 SIESTA | 4.1.5 |
| 3.18 WRF | 4.4.1 |
| 3.19 WRF-Chem | 4.4.1 |
 

### กลุ่ม Application software ที่ต้องโหลด module 
| Application software | Module name|
|--|--|
| abinit | abinit |
| ampl  | ampl | 
| anaconda  | anaconda3 | 
| gromacs | gromacs_gpu |   
| gurobi  | gurobi |  
| julia  | julia | 
| lingo  | lingo | 
| namd  | namd | 
| orca  | orca |
| siesta  | siesta| 
| singularity | singularity |
| PGI Compiler | nvhpc |
| dl_poly | dl_poly_mpi |
| lammps | lammps |
| AMD | aocc |
| Intel | intel |
| Nvidia cuda toolkit | cuda/11.1 <br/> cuda/11.8 <br/> cuda/12.0 |


### ตำแหน่งไฟล์ WRF และ WRF-Chem

    /opt/ohpc/pub/apps/WRF/
    /opt/ohpc/pub/apps/WRF-Chem
    
    ถ้าใช้เวอร์ชั่นที่คอมไพล์ด้วย intel ให้สั่ง module swap gnu9 intel ก่อน
    จากนั้นโหลด netcdf 
    module load netcdf

    ถ้าใช้ intel ต้องสั่งคำสั่งต่อไปนี้ก่อนด้วย
    ulimit -l unlimited
    ulimit -s unlimited
    export KMP_STACKSIZE=20480000000

### OpenFoam ต้องใช้คำสั่ง source
  
    source /opt/ohpc/pub/apps/openfoam/OpenFOAM-10/etc/bashrc


### กลุ่ม Application software ที่ต้องรันผ่าน singularity

- CP2K
- Clara Train SDK


### Compiler ที่ติดตั้งอยู่ในระบบ
| Compiler | GCC | Intel (One API) | AMD (AOCC) | Nvidia PGI | CUDA |
|--|--|--|--|--|--|
| C | gcc | icx | clang | nvc (pgcc)| - |
| C++ | g++ | icpx | clang | nvc++ (pgcc+) | - |
| Fortran | gfortran | ifx | flang | nvfortran (pgfortran) | - |
| Cuda (nvcc) | - | - | - | - | nvcc |
| Module load | - | intel | aocc | nvhpc | cuda/11.1 <br/> cuda/11.8 <br/> cuda/12.0| 

---

# การใช้งาน Erawan Cluster

## วิธีการรีโมท SSH เข้าใช้งานระบบ

เปิด PowerShell บน Windows จากนั้นพิมพ์คำสั่งด้านล่าง

    ssh [Username]@[IP Address or Domain name]

เช่น

    ssh user@erawan.cmu.ac.th

---

## วิธีการคัดลอกไฟล์

    scp /path/to/[ไฟล์ที่ต้องการคัดลอก] [username]@[IP Address or Domain name]:/path/to/[ตำแหน่งที่ต้องการวางไฟล์]

เช่น

    scp C:\temp\test.txt user@erawan.cmu.ac.th:/home/user/

หรือใช้โปรแกรม Filezilla

Download : https://filezilla-project.org/download.php?platform=win64

ระบุ 

- Host : erawan.cmu.ac.th
- Username : user[01-50]
- Password :  
- Port : 22

![enter image description here](https://github.com/somphop26/CMU-Erawan-User/blob/main/imp/Screenshot%20from%202022-12-14%2023-18-32.png?raw=true)


---

## การใช้งาน module environment
- module list เราโหลดอะไรอยู่บ้าง
- module avail มีอะไรให้ใช้บ้าง
- module load โหลดโมดูล 
- module list แสดงโมดูลที่ถูกโหลด
- module unload เลิกโหลดโมดูล
- module swap ใช้กรณีที่โมดูลมีการ conflict กัน
- module purge เลิกโหลดโมดูลทั้งหมด

---


## วิธีการใช้งาน Slurm

Slurm เป็นซอฟต์แวร์ Job scheduler มีหน้าที่ในการจัดลำดับงานในระบบ โดยหลักการทำงานของ Slurm คือผู้ใช้ต้องส่ง Job script ผ่านเครื่อง Login node เข้าไปต่อคิวใน Slurm เพื่อรอที่จะรันงาน เมื่อถึงคิว Slurm จะทำการส่งงานไปรันที่เครื่อง Compute node ตาม Partition ที่ท่านกำหนดในไฟล์ Job script เมื่อประมวลผลเสร็จ ผลลัพธ์จะเก็บอยู่ในตำแหน่งที่ท่านรัน

| Partition | CPU Core | GPU | Node | Time limit |
|--|--:|--:|--|--|
| gpu |     16  | 16 | compute[1-2] | 72 hours |
| cpu |   240  |   | compute[1-2] | 72 hours |
| short | 128  |  8 | compute3     | 24 hours |




ผู้ใช้จะต้องเขียนไฟล์ Job script ขึ้นมาเพื่อส่งงานไปรันที่ Slurm เท่านั้น **ห้ามรีโมทไปรันที่เครื่องโดยตรง** เพราะจะส่งผลให้ระบบมีประสิทธิภาพการทำงานโดยรวมที่ไม่ดี โดยไฟล์ Job script ท่านสามารถระบุความต้องการต่าง ๆ ได้ เช่น ระบุจำนวนทรัพยากรที่ต้องการ (CPU, GPU, RAM) ระบุระยะเวลาที่ใช้ในการรัน ระบุ Partition (Resource group) ที่ต้องการใช้งาน เป็นต้น

1. ssh มายัง login node
2. ให้ submit งานผ่าน Slurm batch หรือ interactive jobs จากเครื่อง login node (ห้ามรันงานที่ login node)
- การ Submit งานที่ใช้ thread ให้กำหนด #SBATCH --cpus-per-task=  ตามจำนวน threads ที่ใช้งาน
- การ Submit งานที่เป็น MPI ให้กำหนด #SBATCH --ntasks=  ตามจำนวน Process ที่ต้องการ 
- การ Submit งานที่ใช้ GPU บางงานใช้ GPU เป็นหลัก ให้กำหนด --cpus-per-task=1 หรือไม่กำหนด เพราะค่า default คือ 1 และขอให้มั่นใจว่าโค้ดของท่านไม่แตก thread

### คำสั่งพื้นฐานสำหรับใช้งาน Slurm มีดังนี้

Submit Job script ไปต่อคิวที่ slurm สำหรับรอประมวลผล

    $ sbatch [Job script file]
        Submitted batch job <jobid>

แสดงข้อมูลเกี่ยวกับงานที่อยู่ในคิวในSlurm

    $ squeue
        JOBID     PARTITION       NAME      USER   ST    TIME  NODES NODELIST(REASON)
      <jobid>           gpu   test.job    user99   R     0:30      1 compute3

ยกเลิกงานที่อยู่ในคิวใน Slurm

    $ scancel [JobID]

แสดงสถานะของ Partition

    $ sinfo
       PARTITION  AVAIL   TIMELIMIT  NODES  STATE  NODELIST
       gpu*          up  3-00:00:00      2   idle compute[1-2]
       cpu           up  3-00:00:00      2   idle compute[1-2]
       short         up  1-00:00:00      1   idle compute3

- Partition คือ กลุ่มประเภทเครื่องที่ใช้งาน (Resource group)
- Timelimit คือ ระยะเวลาสูงสุดที่สามารถใช้งานได้ต่อ 1 งาน
- Nodes     คือ จำนวนเครื่องที่มีอยู่ใน Partition
- Node list คือ ชื่อเครื่อง
- State     คือ สถานะของเครื่อง

| State | ความหมาย |
|--|--|
| idle | สถานะเครื่องว่างไม่มีการใช้งาน |
| alloc | สถานะเครื่องมีการใช้งานทรัพยากรเครื่องเต็ม | 
| mix | สถานะเครื่องมีการใช้งาน CPU บางส่วน |
| down | สถานะเครื่องไม่พร้อมให้ใช้งาน |
| drain | สถานะเครื่องไม่พร้อมให้ใช้งานเนื่องจากเกิดปัญหาภายในระบบ |


แสดงข้อมูลของแต่ละ Compute node

    scontrol show nodes

แสดงข้อมูลของแต่ละงาน

    scontrol show jobs
    
แสดงข้อมูล Partition

    scontrol show partition


### การรัน Slurm ในแบบต่าง ๆ

#### ตัวอย่างการรันงานแบบ Serial Jobs

โดยเนื้อหาจะระบุให้แบ่งงานจำนวน 1 tasks ใช้ CPU ประมวลผลจำนวน 1 core

    #!/bin/bash
    #SBATCH --job-name=mytest        # create a short name for your job
    #SBATCH --nodes=1                # node count
    #SBATCH --ntasks=1               # total number of tasks across all nodes
    #SBATCH --cpus-per-task=1        # cpu-cores per task (>1 if multi-threaded tasks)
    #SBATCH --time=00:01:00          # total run time limit (HH:MM:SS)

    module purge
    Rscript myscript.R

รัน

    sbatch myscriptR.job

แสดงสถานะ

    squeue

![enter image description here](https://github.com/somphop26/CMU-Erawan-User/blob/main/imp/Screenshot%20from%202022-12-21%2017-18-46.png?raw=true)

**JOB STATE CODES (ST)**

Jobs typically pass through several states in the course of their execution. The typical states are PENDING (PD), RUNNING (R), SUSPENDED (S), COMPLETING (CG) , and COMPLETED (CD). An explanation of each state follows.

source https://slurm.schedmd.com/squeue.html#SECTION_JOB-STATE-CODES

---


#### ตัวอย่างการรันงานแบบ Multithreaded Jobs

งานที่เป็น multithreaded ขอให้ระบุพารามิเตอร์ --cpus-per-task= โดยกำหนดจำนวนเท่ากับจำนวนเธรดที่ต้องการแล้วใช้ตัวแปร $SLURM_CPUS_PER_TASK ไประบุในพารามิเตอร์ของคำสั่งดังตัวอย่างต่อไปนี้

    #!/bin/bash
    #SBATCH --job-name=multithread   # create a short name for your job
    #SBATCH --nodes=1                # node count
    #SBATCH --ntasks=1               # total number of tasks across all nodes
    #SBATCH --cpus-per-task=8        # cpu-cores per task (>1 if multi-threaded tasks)
    #SBATCH --time=00:15:00          # maximum time needed (HH:MM:SS)
    
    module load gromacs_gpu
    gmx mdrun -ntomp $SLURM_CPUS_PER_TASK -v -noconfout -nsteps 5000 -s  1536/topol.tpr
    bwa mem -t $SLURM_CPUS_PER_TASK 

รัน 

    sbatch gromac-water.gpu

*** **สำคัญ** ***

ในคำสั่งที่ท่านใช้รัน **ห้ามกำหนด thread ในคำสั่ง** เช่น "gmx mdrun -ntomp **8** -v -noconfout -nsteps 5000 -s  1536/topol.tpr" **ให้กำหนดผ่านตัวแปร Slurm เท่านั้น** เช่น "gmx mdrun -ntomp **$SLURM_CPUS_PER_TASK** -v -noconfout -nsteps 5000 -s  1536/topol.tpr" เพื่อให้ slurm รู้ว่ามีการใช้ thread ไปเท่าไหร่ จะได้จัดสรรงานให้พอดีกับระบบจะได้ไม่เกิด context switching ซึ่งจะส่งผลให้ประสิทธิภาพของระบบโดยรวมไม่ดี




#### ตัวอย่างการรันงานแบบ MPI Jobs


จากเดิมที่รันด้วยมือ จะรันแบบนี้

mpirun with hostfile

    mpirun -np 200 -hostfile hosts ./myprog.o
    
โดยจากเดิมจะกำหนดจำนวน tasks โดยใช้ Option -np 200 เปลี่ยนเป็นระบุในไฟล์ Job script ที่ตัวแปร --ntasks=200 แทน

    #!/bin/bash
    #SBATCH --job-name=mpi-job       # create a short name for your job
    #SBATCH -p cpu                # pritition name
    #SBATCH --ntasks=200             # number of tasks per node
    #SBATCH --cpus-per-task=1        # cpu-cores per task (>1 if multi-threaded tasks)
    #SBATCH --time=00:05:00          # total run time limit (HH:MM:SS)

    module purge
    module load intel
    prun myprog.o

รัน
    
    sbatch mpi.job





#### ตัวอย่างการรันงานแบบ GPU Jobs

ให้ระบุตัวแปร "--gpus=1" เพิ่มขึ้นมาเพื่อกำหนดจำนวน GPU ที่ต้องการ โดยในตัวอย่างระบุให้ใช้ GPU จำนวน 1 การ์ด 

    #!/bin/bash
    #SBATCH --job-name=mnist         # create a short name for your job
    #SBATCH --nodes=1                # node count
    #SBATCH --ntasks=1               # total number of tasks across all nodes
    #SBATCH --cpus-per-task=1        # cpu-cores per task (>1 if multi-threaded tasks)
    #SBATCH --gpus=1                 # total number of GPUs
    #SBATCH --time=01:00:00          # total run time limit (HH:MM:SS)
   
    #CUDA matrix multiplication
    ./mm_optimization

รัน

    sbatch gpu.job

---










## ตัวอย่างการรันซอฟต์แวร์ต่าง ๆ โดย Submit ผ่าน Slurm

### Run python in slurm
source [https://wandb.ai/wandb/common-ml-errors/reports/How-To-Use-GPU-with-PyTorch---VmlldzozMzAxMDk](https://wandb.ai/wandb/common-ml-errors/reports/How-To-Use-GPU-with-PyTorch---VmlldzozMzAxMDk)


สร้างไฟล์สำหรับรัน python

    vi runPytorch.py
    --------------------------------------------
    import torch
    X_train = torch.FloatTensor([0., 1., 2.])
    X_train = X_train.cuda()
    print(X_train)
    
ทดลองรันสคริปต์

    python runPytorch.py
   
**ผลลัพธ์ที่ได้จะ Error เนื่องจากเครื่อง Scheduler node ไม่มี GPU จะต้องส่งคำสั่งไปรันที่เครื่อง compute

สร้างไฟล์ Job script

    vi slurm-pytorch
    ----------------------------------------------
    #!/bin/bash
    #SBATCH --gpus=1                # total number of GPUs
    #SBATCH -p short                # specific partition (compute, memory, gpu)
    #SBATCH -o testpytorch.%j.out   # Name of stdout output file (%j expands to jobId)
    #SBATCH -J testpytorch          # Job name
    #SBATCH -N 1                    # Total number of nodes requested
       
    python runPytorch.py

submit slurm

    sbatch slurm-pytorch



### Submit slurm on Jupyter 

เข้าใช้งานบน web browser ระบุ URL: [http://erawan.cmu.ac.th:8000](http://erawan.cmu.ac.th:8000) แล้ว login เข้าระบบ

![enter image description here](https://github.com/somphop26/CMU-Erawan-User/blob/main/imp/Screenshot%20from%202022-12-13%2021-48-37.png?raw=true)


คลิก + > เลือก Notebook

![enter image description here](https://github.com/somphop26/CMU-Erawan-User/blob/main/imp/jupyter.png?raw=true)


โหลด slurm-magic ก่อนใช้งานคำสั่ง slurm

    %load_ext slurm_magic

  

Submit งานใช้ โดยใช้คำสั่ง %%sbatch แล้วตามด้วย Job script ตามปกติ

    %%sbatch
    #!/bin/bash
    #SBATCH --gpus=1        # total number of GPUs
    #SBATCH -p normal       # specific partition (compute, memory, gpu)
    #SBATCH -o jpjob.%j.out # Name of stdout output file (%j expands to jobId)
    #SBATCH -J jptest       # Job name
    #SBATCH -N 1            # Total number of nodes requested
    
    #CUDA matrix multiplication
    ./mm_optimization

  
ตรวจสอบผลลัพธ์

    cat jpjob.??.out

![enter image description here](https://github.com/somphop26/CMU-Erawan-User/blob/main/imp/Screenshot%20from%202022-12-13%2022-55-24.png?raw=true)
  

Running Jupyter on Slurm GPU Nodes
[https://nero-docs.stanford.edu/jupyter-slurm.html](https://nero-docs.stanford.edu/jupyter-slurm.html)

  

### ตัวอย่างการรัน Singularity
ใช้งาน (รันโดย user)

    module load singularity

ทดลองรันที่เครื่อง compute

    singularity run --nv /opt/ohpc/pub/apps/singularity/cp2k_v9.1.0.sif mpirun -np 1  binder.sh cp2k.psmp -i H2O-dft-ls.NREP2.inp

  
เขียนไฟล์ Job script

    vi runCP2K
    —-------------------------------------------------------------------
    #!/bin/bash
    #SBATCH --gpus=1       # total number of GPUs
    #SBATCH -p short       # specific partition (compute, memory, gpu)
    #SBATCH -o cp2k.%j.out # Name of stdout output file (%j expands to jobId)
    #SBATCH -J cp2kgpu     # Job name
    #SBATCH -N 1           # Total number of nodes requested
    
    #CUDA matrix multiplication
    
    singularity run --nv /opt/ohpc/pub/apps/singularity/cp2k_v9.1.0.sif mpirun -np 1  binder.sh cp2k.psmp -i H2O-dft-ls.NREP2.inp

  
รัน Job script ที่เครื่อง erawan

    sbatch runCP2K

### OpenFoam Example

ใช้ enviroment

    source /opt/ohpc/pub/apps/openfoam/OpenFOAM-10/etc/bashrc

คัดลอกสคริปต์ตัวอย่าง

    cp -r /opt/ohpc/pub/apps/openfoam/OpenFOAM-10/tutorials/incompressible/icoFoam/cavity ~/openfoam

สร้างไฟล์ job script

    vi slurm-openfoam.sh 
    ----------------------------------------------------------------------------
    #!/bin/bash
    #SBATCH -J openfoam           # Job name
    #SBATCH -o jobopenfoam.%j.out # Name of stdout output file (%j expands to jobId)
    #SBATCH -N 1                  # Total number of nodes requested
    #SBATCH -n 8                  # Total number of mpi tasks requested
    #SBATCH -t 01:30:00           # Run time (hh:mm:ss) - 1.5 hours

    ~/openfoam/Allrun
    cd ~/openfoam/cavity/
    blockMesh
    icoFoam


รัน

    sbatch slurm-openfoam.sh



### Run Jupyter Notebook 

**วิธีนี้จะใช้เฉพาะกรณีที่ท่านต้องการทดสอบสคริปต์ Python ของท่านเท่านั้นเมื่อใช้เสร็จหรือเลิกใช้งานควร สั่ง "scancel [JOBID]" เพื่อให้ระบบคืนทรัพยากรเนื่องจากวิธีด้านล่างนี้จะจองทรัพยากรไว้ตามระยะเวลาที่ท่านกำหนด

โหลดโมดูล anaconda3

    module load anaconda3
    
สร้าง enviroment ของท่าน

    conda create -n [enviroment name]
    conda init bash 
    conda config --set auto_activate_base False #กำหนด ให้ไม่ auto activate base environment
    
เช้าใช้งาน enviroment

    conda activate [enviroment name]
    
ติดตั้ง jupyterlib ใน enviroment

    conda install -c conda-forge jupyterlab


### Running Jupyter on Slurm GPU Nodes
https://nero-docs.stanford.edu/jupyter-slurm.html

สร้างสคริปต์ jupyter.job

    #!/bin/bash
    #SBATCH --job-name=jupyter
    #SBATCH --gpus=1
    #SBATCH --time=02:00:00
 
    source /home/${USER}/.bashrc
    conda activate [enviroment name]
    cat /etc/hosts
    jupyter lab --ip=0.0.0.0 --port=8888

** พอร์ต 8888 คือกำหนดว่าให้ jupyterlab รันที่พอร์ตไหน ให้เปลี่ยนไม่ให้ซ้ำ ตั้งสูง ๆ ไว้เพราะพอร์ตหมายเลขน้อย ๆ อาจจะไปชนกับ service อื่น ๆ โดยเฉพาะที่ต่ำกว่า 1024 

submit

    sbatch jupyter.job

เมื่อ Job รัน (R) แล้ว  ให้ดู output ว่าไปรันที่เครื่องไหน

ทำ ssh tunnel ไปยังพอร์ตที่เรากำหนดให้ Jupyter ตอน submit job

    ssh -L 9999:10.98.4.XX:8888 cmu@erawan.cmu.ac.th

ข้างบนเป็นการกำหนดให้ Local port 9999 เชื่อมไปยังเครื่อง 10.98.4.XX:8888

*อ่านเพิ่มเติม https://www.tunnelsup.com/how-to-create-ssh-tunnels

เปิดหน้าเว็บไปที่ http://localhost:9999 เข้าไปดู token ใน output slurm





