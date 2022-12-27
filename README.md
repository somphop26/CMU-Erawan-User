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

Compute node Local disk **scratch**

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
| 3.5 LAMMPS | 20210527 |
| 3.6 LINGO | 19 |
| 3.7 Quantum Espresso | 6.8 |
| 3.8 Singularity | 3.7.1-5.1.ohpc.2.1 |
| 3.9 ABINIT | 9.6.2 |
| 3.10 CP2K | 9.1.0 |
| 3.11 DL_POLY | 4 |tate 
| 3.12 FreeSurfer | 7.3.2 |
| 3.13 NAMD | 2.14 |
| 3.14 NWChem | 7.0.2 |
| 3.15 OpenFOAM | 10 |
| 3.16 ORCA | 5.0.3 |
| 3.17 SIESTA | 4.1.5 |
| 3.18 WRF | 4.4.1 |
| 3.19 WRF-Chem | 4.4.1 |
 

### Application software ที่ต้องโหลด module 
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


### WRF และ WRF-Chem

    ตำแหน่งไฟล์ 
    /opt/ohpc/pub/apps/WRF/
    /opt/ohpc/pub/apps/WRF-Chem
    
    ถ้าใช้เวอร์ชั่นที่คอมไพล์ด้วย intel ให้สั่ง 
    module swap gnu9 intel 
    จากนั้นโหลด netcdf 
    module load netcdf

    ถ้าใช้ intel ต้องสั่งคำสั่งต่อไปนี้ก่อนด้วย
    ulimit -l unlimited
    ulimit -s unlimited
    export KMP_STACKSIZE=20480000000

### OpenFoam ต้องใช้คำสั่ง source
  
    source /opt/ohpc/pub/apps/openfoam/OpenFOAM-10/etc/bashrc


### Application software ที่ต้องรันผ่าน singularity

- CP2K
- Clara Train SDK


### Compiler ที่ติดตั้งอยู่ในระบบ
| Compiler | GCC | Intel (One API) | AMD (AOCC) | Nvidia PGI | CUDA |
|--|--|--|--|--|--|
| C | gcc | icx | clang | nvc (pgcc)| - |
| C++ | g++ | icpx | clang | nvc++ (pgcc+) | - |
| Fortran | gfortran | ifx | flang | nvfortran (pgfortran) | - |
| Cuda (nvcc) | - | - | - | - | nvcc |
| Module load | gnu9 | intel | aocc | nvhpc | cuda/11.1 <br/> cuda/11.8 <br/> cuda/12.0| 

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
- Username : user
- Password :  
- Port : 22

![enter image description here](https://github.com/somphop26/CMU-Erawan-User/blob/main/imp/Screenshot%20from%202022-12-14%2023-18-32.png?raw=true)


---

## การใช้งาน Module environment

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

ในระบบได้มีการแบ่ง Partition (Resource group) ดังนี้
| Partition | CPU Core | GPU | Node | Time limit |
|--|--:|--:|--|--|
| gpu |     64  | 16 | compute[1-2] | 72 hours |
| cpu |   192  |   | compute[1-2] | 72 hours |
| short | 128  |  8 | compute3     | 24 hours |


ขั้นตอนการใช้งาน
1. ssh มายัง login node
2. เขียนไฟล์ Job script เพื่อส่งงานไปรันที่ Slurm (**ห้ามรีโมทไปรันที่เครื่อง Compute node โดยตรง**)
3. ทำการ submit งานผ่าน Slurm batch หรือ interactive jobs จากเครื่อง login node (**ห้ามรันที่เครื่อง login node**)
- การ Submit งานที่ใช้ thread ให้กำหนด #SBATCH --cpus-per-task=  ตามจำนวน threads ที่ใช้งาน
- การ Submit งานที่เป็น MPI ให้กำหนด #SBATCH --ntasks=  ตามจำนวน Process ที่ต้องการ 
- การ Submit งานที่ใช้ GPU เป็นหลัก ที่ partition GPU ให้กำหนด --cpus-per-task ตามความเหมาะสม หากงานไม่ได้แตก thread ให้กำหนดเป็น 1 หากแตก thread กำหนดได้ระหว่าง 2-4 threads เพื่อแบ่งปันคอร์ที่เหลือกับการ์ดใบอื่น ๆ จะได้ใช้งานระบบได้อย่างเต็มความสามารถ 


### คำสั่งพื้นฐานสำหรับใช้งาน Slurm มีดังนี้

Submit Job script ไปต่อคิวที่ Slurm เพื่อรอประมวลผล

    $ sbatch [Job script file]
        Submitted batch job <jobid>

แสดงข้อมูลคิวงานทั้งหมด

    $ squeue
        JOBID     PARTITION       NAME      USER   ST    TIME  NODES NODELIST(REASON)
      <jobid>           gpu   test.job    user99   R     0:30      1 compute3

JOB STATE CODES (ST) โดยทั่วไปงานจะผ่านหลายสถานะในระหว่างการดำเนินการ สถานะทั่วไปคือ PENDING (PD), RUNNING (R), SUSPENDED (S), COMPLETING (CG) และ COMPLETED (CD) 
อ่านเพิ่มเติมที่ https://slurm.schedmd.com/squeue.html#SECTION_JOB-STATE-CODES

JOB REASON CODE (NODELIST(REASON)) จะแสดงรายละเอียดสถานะต่าง ๆ ของแต่ละคิว เช่น
- หากในคิวนั้นมีการประมวลผลจะขึ้นชื่อเครื่องที่ใช้ประมวลผล compute1..3
- เมื่อมีการขอทรัพยากรเกินกว่าจำนวนที่มีอยู่ใน Partition จะขึ้นสถานะ PartitionConfig
- อยู่ระหว่างรอทรัพยากรว่างเพียงพอต่องานที่รันจะขึ้นสถานะ Resources
- อยู่ระหว่างรอคิวโดยมีจัดลำดับคิวตาม scheduling algorithm (SchedulerType=sched/backfill) จะขึ้นสถานะ Priority

อ่านเพิ่มเติมที่ : https://slurm.schedmd.com/squeue.html#SECTION_JOB-REASON-CODES

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

---

### การรัน Slurm ในแบบต่าง ๆ

ในไฟล์ job script ของ Slurm ควรระบุ option ดังนี้

- #SBATCH --time=[ระยะเวลาที่ใช้รันงาน] <br/> มีผลต่อการจัดลำดับความสำคัญของงาน เพื่อไม่ให้ CPU-core ว่างในระหว่างที่มีคิวรอ CPU ให้เพียงพอกับงานนั้น ตัวจัดลำดับงานจะคำนวณว่าสามารถหยิบงานใดที่สามารถนำมารันก่อนโดยที่ไม่กระทบกับงานที่รอคิวอยู่ หากมีตัวจัดลำดับงานก็จะหยิบงานนั้นมารันก่อนแทนที่จะปล่อยให้ CPU-core ว่างอยู่เฉย ๆ <br/> ท่านควรระบุเวลาให้ใกล้เคียงความเป็นจริงที่ท่านใช้รันที่สุดเพื่อระบบจะได้จัดสรรลำดับความสำคัญให้ท่าน หากไม่ระบุค่าเริ่มต้นจะเป็นค่า Timelimit ของ Partition ที่ท่านใช้งาน 
-  #SBATCH -p [ชื่อพาติชัน] <br/> กำหนดพาร์ติชันที่ต้องการใช้งาน โดยแต่ละพาร์ติชันมีทรัพยากร และระยะเวลาที่จำกัดแตกต่างกัน ผู้ใช้ควรดูว่างานของท่านใช้ทรัพยากร และระยะเวลาเท่าไหร่ที่เหมาะสมกับพาร์ติชันที่จะระบุ 

Option ที่สำคัญนอกจากนี้ได้อธิบายในตัวอย่างการรัน Slurm ในแบบต่าง ๆ ดังนี้

#### แบบ Serial Jobs

งานที่เป็น Serial Job จะใช้ CPU-core เดียว ให้กำหนดตัวแปร "--ntasks=1" ซึ่งจะกำหนดหรือไม่ก็ได้เพราะค่าเริ่มต้นของ --ntasks=1 อยู่แล้ว


    #!/bin/bash
    #SBATCH --job-name=mytest        # create a short name for your job
    #SBATCH --nodes=1                # node count
    #SBATCH --ntasks=1               # total number of tasks across all nodes
    #SBATCH --cpus-per-task=1        # cpu-cores per task (>1 if multi-threaded tasks)
    #SBATCH --time=00:20:00          # total run time limit (HH:MM:SS)

    module purge
    Rscript myscript.R

รัน

    sbatch myscriptR.job

---


#### แบบ Multithreaded Jobs

งานที่เป็น Multithreaded ขอให้ระบุพารามิเตอร์ --cpus-per-task= โดยกำหนดจำนวนเท่ากับจำนวนเธรดที่ต้องการแล้วใช้ตัวแปร $SLURM_CPUS_PER_TASK ไประบุในพารามิเตอร์ของคำสั่ง 

ดังตัวอย่างต่อไปนี้

    #!/bin/bash
    #SBATCH --job-name=multithread   # create a short name for your job
    #SBATCH --nodes=1                # node count
    #SBATCH --ntasks=1               # total number of tasks across all nodes
    #SBATCH --cpus-per-task=8        # cpu-cores per task (>1 if multi-threaded tasks)
    #SBATCH --time=00:15:00          # maximum time needed (HH:MM:SS)
    
    # ตัวอย่างคำสั่งการรันซอฟต์แวร์ Gromacs แบบ Multithreaded Job
    module load gromacs_gpu
    gmx mdrun -ntomp $SLURM_CPUS_PER_TASK -v -noconfout -nsteps 5000 -s  1536/topol.tpr
    bwa mem -t $SLURM_CPUS_PER_TASK 

รัน 

    sbatch gromac-water.gpu


*** **สำคัญ** ***

ควรกำหนดค่าพารามิเตอร์ **--cpus-per-task=** และกำหนด CPU-core ในคำสั่งโดยใช้ตัวแปร **$SLURM_CPUS_PER_TASK** แทนการกำหนดเป็นตัวเลข การกำหนดเช่นนี้จะทำให้ Slurm รู้ว่างานของท่านใช้ CPU-core เท่าไหร่จะได้จัดสรรได้ถูกต้อง เพื่อจะได้ไม่มีปัญหาการรันงานเกินกว่าจำนวน CPU-core ที่มีในระบบซึ่งจะส่งผลต่อประสิทธิภาพการใช้งาน
 
---

#### แบบ MPI Jobs

จากเดิมการรัน mpi จะรันคำสั่งด้านล่าง ซึ่งจะกำหนดจำนวน tasks โดยใช้ Option -np [ตามด้วยจำนวน tasks ที่ต้องการ]

    mpirun -np 192 -hostfile hosts ./myprog.o
    
เมื่อ Submit ผ่าน Slurm ให้กำหนดที่ตัวแปร "--ntasks=[จำนวน tasks ที่ต้องการ]" และใช้คำสั่ง prun แทนการสั่งแบบเดิม

    #!/bin/bash
    #SBATCH --job-name=mpi-job       # create a short name for your job
    #SBATCH -p cpu                   # pritition name
    #SBATCH --ntasks=192             # number of tasks per node
    #SBATCH --cpus-per-task=1        # cpu-cores per task (>1 if multi-threaded tasks)
    #SBATCH --time=00:20:00          # total run time limit (HH:MM:SS)

    module purge
    module load intel
    prun myprog.o

รัน
    
    sbatch mpi.job

---

#### แบบ GPU Jobs

งานที่ใช้ GPU ให้ระบุตัวแปร "--gpus=1" โดยในตัวอย่างระบุให้ใช้ GPU จำนวน 1 การ์ด 

    #!/bin/bash
    #SBATCH --job-name=mnist         # create a short name for your job
    #SBATCH --nodes=1                # node count
    #SBATCH --ntasks=1               # total number of tasks across all nodes
    #SBATCH --cpus-per-task=4        # cpu-cores per task (>1 if multi-threaded tasks)
    #SBATCH --gpus=1                 # total number of GPUs
    #SBATCH --time=01:00:00          # total run time limit (HH:MM:SS)
   
    #CUDA matrix multiplication
    ./mm_optimization

รัน

    sbatch gpu.job

---


### การสร้างสภาพแวดล้อม conda สำหรับงานที่ใช้ Python

สร้างสภาพแวดล้อม python (ในตัวอย่างสร้างสภาพแวดล้อมชื่อ test)

    module load anaconda3
    conda create -n test python=3.7

กำหนดไม่ให้ค่าเริ่มต้นใช้งานสภาพแวดล้อม base (สั่งครั้งเดียว) และกำหนด shell ในสภาพแวดล้อมเป็น bash จากนั้น logout และ Login ใหม่

    conda config --set auto_activate_base False
    conda init bash
    exit
    ssh [username]@erawan.cmu.ac.th

เข้าใช้งานสภาพแวดล้อมที่สร้าง 

    conda activate test
    python --version
    
ติดตั้ง software ในสภาพแวดล้อม (ในตัวอย่างทำการติดตั้ง numpy)
    
    conda install numpy
    หรือ
    pip3 install numpy

หากต้องการลบสภาพแวดล้อมใช้คำสั่ง (ในตัวอย่างลบสภาพแวดล้อมชื่อ test)

    conda env remove -n test

---

### การ Submit Slurm บน Jupyterhub

เข้าใช้งานบน web browser ระบุ URL: [https://erawan.cmu.ac.th:8000](https://erawan.cmu.ac.th:8000) แล้ว login เข้าระบบ

![enter image description here](https://github.com/somphop26/CMU-Erawan-User/blob/main/imp/Screenshot%20from%202022-12-26%2010-50-19.png?raw=true)


คลิก + > เลือก Notebook

![enter image description here](https://github.com/somphop26/CMU-Erawan-User/blob/main/imp/Screenshot%20from%202022-12-26%2010-51-17.png?raw=true)


โหลด slurm-magic ก่อนใช้คำสั่ง Slurm

    %load_ext slurm_magic


Submit งานใช้ โดยใช้คำสั่ง %%sbatch แล้วตามด้วย Job script ตามปกติ

    %%sbatch
    #!/bin/bash
    #SBATCH --gpus=1                 # total number of GPUs    
    #SBATCH -p gpu                   # specific partition (compute, memory, gpu)
    #SBATCH -o mytest.%j.out         # Name of stdout output file (%j expands to jobId)
    #SBATCH --job-name=mytest        # Job name
    #SBATCH --time=10:00:00 
    
    # ตัวอย่างการรันโค้ดไพธอนด้วยใช้สภาพแวดล้อม conda
    source /home/${USER}/.bashrc
    conda activate test
    python program.py
    python --version


![enter image description here](https://github.com/somphop26/CMU-Erawan-User/blob/main/imp/Screenshot%20from%202022-12-27%2010-08-07.png?raw=true)


### การรัน Singularity บน slurm

เขียนไฟล์ Job script

    vi runCP2K
    —-------------------------------------------------------------------
    #!/bin/bash
    #SBATCH --gpus=1                # total number of GPUs
    #SBATCH --job-name= cp2kgpu     # create a short name for your job
    #SBATCH -p gpu                  # specific partition (compute, memory, gpu)
    #SBATCH -o cp2k.%j.out          # Name of stdout output file (%j expands to jobId)
    #SBATCH --ntasks=200            # number of tasks per node
    #SBATCH --time=05:00:00 
    
    # ตัวอย่างคำสั่งการรันซอฟต์แวร์ CP2K ผ่าน singularity
    singularity run --nv /opt/ohpc/pub/apps/singularity/cp2k_v9.1.0.sif prun  binder.sh cp2k.psmp -i H2O-dft-ls.NREP2.inp

  
รัน Job script ที่เครื่อง erawan

    sbatch runCP2K

