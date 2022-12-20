
# Erawan Cluster
## Hardware Spec 
1 Scheduler node

3 Compute with 8 GPU Nvidia A100 (128 Cores, 2TB)

Storage

-   Archive 288TB (Usable)
-   Parallel File System (wekaio) 500TB

Network

-   25Gbps
-   200Gbps Infiniband

     
## Software
-   OS: Rocky-8.7
-   OpenHPC 2.4


## Nework share directory (via NFS)
- /home 288TB Read Write
- /opt/ohpc/pub Read only

### เมื่อ wekaio (Raw 500TB) Raw
- /proj แล้วแต่กำหนดให้กับผู้ใช้
- /sharedata/blast/db 100TB
- /scratch 100TB ลบทุก 60 วัน


### NCBI blast database
ตอนนี้ /home/sharedata/blast/db

Compute node Local disk

/scratch.local
 
## List of application software
- 2.1 Python 3.6.8
- 2.2 TensorFlow 2.6.2
- 2.3 Anaconda 3-2022.05
- 2.4 Keras 2.6.0
- 2.5 Pytorch 1.10.2+cu113
- 2.6 openCV 3.4.6
- 2.7 R program 4.2.2
- 2.8 Transformers 4.18.0
- 2.9 AMPL 20221023
- 2.10 C language 9.4.0 , 8.5.0
- 2.11 Clara Train SDK 4.1
- 2.12 CUDA Toolkit 11.8
- 2.13 CuDNN 8.7
- 2.14 GCC 9.4.0 , 8.5.0
- 2.15 GNU C++ 9.4.0 , 8.5.0
- 2.16 Matplotlib 3.0.3
- 2.17 NumPy 1.19.5 , 1.14.3
- 2.18 Open MPI 4.1.1 (gcc) , 4.1.4 (intel)
- 2.19 pandas 0.25.3
- 2.20 PGI Compiler (NVHPC-2022) 22.11
- 2.21 Ray 2.1.0
- 2.22 Julia 1.8.3
- 2.23 Mkl (bundle with Intel One API) 2022.2.1
 
- 3.1 Jupyter notebook 1.13.5
- 3.2 Gurobi 10.0
- 3.3 GROMACS 2019.6
- 3.4 BLAST 2.13.0
- 3.5 LAMMPS 20190807
- 3.6 LINGO 19
- 3.7 Quantum Espresso 6.8
- 3.8 Singularity 3.7.1-5.1.ohpc.2.1
- 3.9 ABINIT 9.6.2
- 3.10 CP2K
- 3.11 DL_POLY 1.10-12
- 3.12 FreeSurfer 7.3.2
- 3.13 NAMD 2.14
- 3.14 NWChem 7.0.2
- 3.15 OpenFOAM 10
- 3.16 ORCA 5.0.3
- 3.17 SIESTA 4.1.5
- 3.18 WRF 4.4.1
- 3.19 WRF-Chem 4.4.1
  
#Share space for application data

#NCBI blast database

#ตอนนี้ /home/sharedata/blast/db หลังจากระบบเรียบร้อยแล้วจะเปลี่ยนเป็น /sharedata/blast/db โดยตอนนี้มีข้อมูลต่อไปนี้วางอยู่แล้ว
-   nr
-   nt
-   refseq_protein
-   refseq_rna
-   swissprot


## กลุ่ม Application software ที่ต้องโหลด module 
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

## กลุ่ม Application software ที่ต้องโหลด module อื่นที่เกี่ยวข้อง
- MKL ใช้โมดูล intel
- WRF ใช้โมดูล intel, netcdf
- WRF-Chem ใช้โมดูล intel, netcdf

ตำแหน่งไฟล์ WRF

    /opt/ohpc/pub/apps/WRF/intel/

## กลุ่ม Application software ที่ต้องสั่ง source
OpenFoam
    
คำสั่ง source

    source /opt/ohpc/pub/apps/openfoam/OpenFOAM-10/etc/bashrc



## กลุ่ม Application software ที่ต้องรันผ่าน singularity
- CP2K
- Clara Train SDK


## วิธีการรีโมท
เปิด PowerShell บน Windows จากนั้นพิมพ์คำสั่งด้านล่าง

    ssh  [Username]@[IP Address or Domain name]

เช่น

    ssh user@erawan.cmu.ac.th

## วิธีการคัดลอกไฟล์

    scp /path/to/[ไฟล์ที่ต้องการคัดลอก] [username]@[IP Address or Domain name]:/path/to/[ตำแหน่งที่ต้องการวางไฟล์]

เช่น

    scp C:\temp\test.txt user@10.110.0.11:/home/user/

หรือใช้โปรแกรม Filezilla

Download : https://filezilla-project.org/download.php?platform=win64

ระบุ 

- Host : erawan.cmu.ac.th
- Username : user[01-50]
- Password :  
- Port : 22

![enter image description here](https://github.com/somphop26/CMU-Erawan-User/blob/main/imp/Screenshot%20from%202022-12-14%2023-18-32.png?raw=true)


## การใช้งาน module environment
- module list เราโหลดอะไรอยู่บ้าง
- module avail มีอะไรให้ใช้บ้าง
- module load โหลดโมดูล 
- module list แสดงโมดูลที่ถูกโหลด
- module unload เลิกโหลดโมดูล
- module swap ใช้กรณีที่โมดูลมีการ conflict กัน
- module purge เลิกโหลดโมดูลทั้งหมด

## ตัวอย่างการรัน Slurm Serial
Create file "list.R"

    sum(0:9)
    append(LETTERS[1:13],letters[14:26])
    c(1,6,4,9)*2
    something <- c(1,4,letters[2])
    length(something)

Test R Script

    Rscript list.R


Create file Job script "test.job"


    #!/bin/sh 
    #SBATCH -p normal
    #SBATCH -J mytest
    #SBATCH -o job.%j.out
    
    Rscript list.R


| พารามิเตอร์ | คำอธิบาย |
|--|--|
| #SBATCH -p **[partition name]** | ระบุพาร์ติชันที่ต้องการใช้งาน |
| #SBATCH -J **[job name]** | ระบุชื่องาน |
| #SBATCH -o **[output name]** | ระบุชื่อไฟล์ผลลัพธ์ |
| #SBATCH -N **[number of node]** | ระบุจำนวนเครื่อง (nodes) ที่ต้องการใช้งาน |
| #SBATCH -t **[time]** | ระบุระยะเวลาที่ใช้จำกัดในการรัน รูปแบบ ชั่วโมง:นาที:วินาที |
| #SBATCH --gpus=**[number of GPU]** | ระบุจำนวน GPU ที่ใช้ |

Run

    sbatch test.job

View status

    squeue

![enter image description here](https://raw.githubusercontent.com/somphop26/CMU-Erawan-User/main/imp/Screenshot%20from%202022-12-13%2022-16-00.png)

Cancel job

    scancel <jobid>

---
**JOB STATE CODES**

Jobs typically pass through several states in the course of their execution. The typical states are PENDING, RUNNING, SUSPENDED, COMPLETING, and COMPLETED. An explanation of each state follows.

**BF BOOT_FAIL**

Job terminated due to launch failure, typically due to a hardware failure (e.g. unable to boot the node or block and the job can not be requeued).

**CA CANCELLED**

Job was explicitly cancelled by the user or system administrator. The job may or may not have been initiated.

**CD COMPLETED**

Job has terminated all processes on all nodes with an exit code of zero.

**CF CONFIGURING**

Job has been allocated resources, but are waiting for them to become ready for use (e.g. booting).

**CG COMPLETING**

Job is in the process of completing. Some processes on some nodes may still be active.

**DL DEADLINE**

Job terminated on deadline.

**F FAILED**

Job terminated with non-zero exit code or other failure condition.

**NF NODE_FAIL**

Job terminated due to failure of one or more allocated nodes.

**OOM OUT_OF_MEMORY**

Job experienced out of memory error.

**PD PENDING**

Job is awaiting resource allocation.

**PR PREEMPTED**

Job terminated due to preemption.

**R RUNNING**

Job currently has an allocation.

**RD RESV_DEL_HOLD**

Job is being held after requested reservation was deleted.

**RF REQUEUE_FED**

Job is being requeued by a federation.

**RH REQUEUE_HOLD**

Held job is being requeued.

**RQ REQUEUED**

Completing job is being requeued.

**RS RESIZING**

Job is about to change size.

**RV REVOKED**

Sibling was removed from cluster due to other cluster starting the job.

**SI SIGNALING**

Job is being signaled.

**SE SPECIAL_EXIT**

The job was requeued in a special state. This state can be set by users, typically in EpilogSlurmctld, if the job has terminated with a particular exit value.

**SO STAGE_OUT**

Job is staging out files.

**ST STOPPED**

Job has an allocation, but execution has been stopped with SIGSTOP signal. CPUS have been retained by this job.

**S SUSPENDED**

Job has an allocation, but execution has been suspended and CPUs have been released for other jobs.

**TO TIMEOUT**

Job terminated upon reaching its time limit.

source https://slurm.schedmd.com/squeue.html#SECTION_JOB-STATE-CODES

---

## ตัวอย่างการรัน MPI
Run a Test Job

create mpi programming "myrank.c"
   

    #include <stdio.h>
    #include "mpi.h"
    int main(int argc,char *argv[]) {
             int size,len,rank;
             char procname[100];
             MPI_Init(&argc,&argv);
             MPI_Comm_size(MPI_COMM_WORLD,&size);
             MPI_Comm_rank(MPI_COMM_WORLD,&rank);
             MPI_Get_processor_name(procname,&len);
             printf("I'm rank = %d of %d process on %s\n",rank,size,procname);
             MPI_Finalize();
             return 0;
    }

Compile

    mpicc myrank.c

Run

    ./a.out

MPI Run

    mpirun -np 2 ./a.out

Create hostfile (for openmpi) 

    nano hosts
    -------------------
    compute0 slots=128
    compute1 slots=128
    compute2 slots=128

mpirun with hostfile

    mpirun -np 4 -hostfile hosts ./a.out



## ตัวอย่างการรัน Slurm Multi-thread
Batch execution
Copy example job script 

    cp /opt/ohpc/pub/examples/slurm/job.mpi .

Examine contents (and edit to set desired job sizing characteristics)
Edit file job script "job.mpi"

    #!/bin/bash
    #SBATCH -J test            # Job name
    #SBATCH -o job.%j.out      # Name of stdout output file (%j expands to jobId)
    #SBATCH -N 1               # Total number of nodes requested
    #SBATCH -n 8               # Total number of mpi tasks requested
    #SBATCH -t 01:30:00        # Run time (hh:mm:ss) - 1.5 hours
    
    # Launch MPI-based executable
    prun ./a.out

Submit job for batch execution

    sbatch job.mpi


รายละเอียด slurm เพิ่มเติม [https://thaisc.io/คู่มือผู้ใช้งาน/](https://thaisc.io/%E0%B8%84%E0%B8%B9%E0%B9%88%E0%B8%A1%E0%B8%B7%E0%B8%AD%E0%B8%9C%E0%B8%B9%E0%B9%89%E0%B9%83%E0%B8%8A%E0%B9%89%E0%B8%87%E0%B8%B2%E0%B8%99/)


## ตัวอย่างการรัน Slurm ใช้ GPU ประมวลผล
ดาวน์โหลดไฟล์ซอร์สโค้ดสำหรับทดสอบ

    wget https://gist.githubusercontent.com/leimao/bea971e07c98ce669940111b48a4cd7b/raw/f55b4dbf6c51df6b3604f2b598643f9672251f7b/mm_optimization.cu
    
ทำการคอมไพล์ซอฟต์แวร์

    module load nvhpc
    nvcc mm_optimization.cu -o mm_optimization

สร้างไฟล์ Job Script

    vi gpu_job.sh
    --------------------------------------------------------------
    #!/bin/bash
    #SBATCH --gpus=1           # total number of GPUs
    #SBATCH -o gpujob.%j.out   # Name of stdout output file (%j expands to jobId)
    #SBATCH -J gputest         # Job name
    #SBATCH -N 1               # Total number of nodes requested
    #SBATCH -t 01:00:00        # Run time (hh:mm:ss) - 1 hours

    
    #CUDA matrix multiplication
    date
    ./mm_optimization
    date    

รัน

    sbatch gpu_job.sh
    squeue

ตรวจสอบผลลัพธ์

    cat gpujob.<jobid>.out


## Run python in slurm
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



## Jupyter notebook

เข้าใช้งานบน web browser ระบุ URL: [http://10.110.0.11:8000](http://10.110.0.11:8000/) หรือ [http://erawan.cmu.ac.th:8000](http://erawan.cmu.ac.th:8000) แล้ว login เข้าระบบ

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

  

## ตัวอย่างการรัน Singularity
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

  

## Gromacs Example
ใช้ตัวอย่างจากลิงค์นี้ 
https://catalog.ngc.nvidia.com/orgs/hpc/containers/gromacs

gromac on GPU

    wget https://ftp.gromacs.org/pub/benchmarks/water_GMX50_bare.tar.gz
    tar xvf water_GMX50_bare.tar.gz
    cd ./water-cut1.0_GMX50_bare/1536

ทดลองรันที่เครื่อง

    module load gromacs_gpu
    gmx grompp -f pme.mdp

สร้างไฟล์ Job script

    vi gromac-water.gpu
    --------------------------------------------------------------
    #!/bin/bash
    #SBATCH --gpus=1              # total number of GPUs
    #SBATCH -p short              # specific partition (compute, memory, gpu)
    #SBATCH -o gromacs.%j.out     # Name of stdout output file (%j expands to jobId)
    #SBATCH --cpus-per-task=8
    
    module load gromacs-gpu
    gmx mdrun -nt $SLURM_CPUS_PER_TASK -v -noconfout -nsteps 5000 -s  topol.tpr

รัน 

    sbatch gromac-water.gpu


## OpenFoam Example

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


