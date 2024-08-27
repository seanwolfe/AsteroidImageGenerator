from mpi4py import MPI

def main():
    # Initialize MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    # Print rank and size to confirm process initialization
    print(f"Process {rank} out of {size} processes initialized")

    if rank == 0:
        # Single process section 1
        print("Single process section 1 - Rank 0 only")

    # Synchronize processes
    print(f"Process {rank} before first barrier")
    comm.Barrier()
    print(f"Process {rank} after first barrier")

    # Multi process section
    print(f"Process {rank} entering multi-process section")

    # Synchronize processes
    print(f"Process {rank} before second barrier")
    comm.Barrier()
    print(f"Process {rank} after second barrier")

    if rank == 0:
        # Single process section 3
        print("Single process section 3 - Rank 0 only")

if __name__ == "__main__":
    main()