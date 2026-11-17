!=======================================================================!
! Copyright (c) 2009-2026 Hans Pabst                                    !
! Copyright (c) 2009-2026 Intel Corporation                             !
! This file is part of the LIBXSTREAM library.                          !
!                                                                       !
! For information on the license, see the LICENSE file.                 !
! Further information: https://github.com/hfp/libxstream/               !
! SPDX-License-Identifier: BSD-3-Clause                                 !
!=======================================================================!

! DBCSR's G2G multiplication (device buffers handed to MPI, block norms
! on the device) against its host algorithm, on what is least regular:
! block extents from 1 to the limit of the SMM kernels, norms over many
! orders of magnitude for filtering, an uneven distribution, either
! transpose, and every product repeated. G2G aborts on any stack the
! device does not take: a larger block, or a stack of mixed shapes, as
! DBCSR collects shapes beyond NSTACKS per extent (at most 5) into one.
! A block filtered differently changes the result by less than EPS, a
! wrong norm by far more than TOLERANCE.
      PROGRAM dbcsr_g2g
        USE :: MPI
        USE :: dbcsr_api, ONLY: dbcsr_type, dbcsr_distribution_type,    &
     &    dbcsr_init_lib, dbcsr_finalize_lib, dbcsr_set_config,         &
     &    dbcsr_distribution_new, dbcsr_distribution_release,           &
     &    dbcsr_create, dbcsr_release, dbcsr_put_block, dbcsr_finalize, &
     &    dbcsr_get_stored_coordinates, dbcsr_multiply, dbcsr_checksum, &
     &    dbcsr_get_num_blocks, dbcsr_print_statistics,                 &
     &    dbcsr_type_no_symmetry, dbcsr_type_real_8
        IMPLICIT NONE

        INTEGER, PARAMETER :: DP = KIND(0D0)
        INTEGER, PARAMETER :: NSIZES = 5, NBLKS = 27, NREPEAT = 3
        INTEGER, PARAMETER :: SIZES(NSIZES) = (/ 1, 13, 23, 64, 80 /)
        ! not a round number: a product of norms meets it by chance only
        REAL(DP), PARAMETER :: EPS = 3.14159D-9, TOLERANCE = 1D-6
        CHARACTER(LEN=1), PARAMETER :: TRANS(2) = (/ 'N', 'T' /)
        TYPE(dbcsr_distribution_type) :: dist
        TYPE(dbcsr_type) :: a, b
        INTEGER, POINTER :: bsize(:), rdist(:), cdist(:)
        INTEGER :: comm, ierr, rank, nranks, io, nfail, i, r, ta, mode
        INTEGER :: dims(2)
        LOGICAL :: periods(2)
        ! unfiltered (0), host (1), and G2G (2): blocks and checksums
        INTEGER :: nb(0:2)
        REAL(DP) :: cs(2, 0:2)

        CALL MPI_INIT(ierr)
        CALL MPI_COMM_SIZE(MPI_COMM_WORLD, nranks, ierr)
        CALL MPI_COMM_RANK(MPI_COMM_WORLD, rank, ierr)
        dims = 0
        periods = .TRUE.
        CALL MPI_DIMS_CREATE(nranks, 2, dims, ierr)
        CALL MPI_CART_CREATE(MPI_COMM_WORLD, 2, dims, periods, .FALSE., &
     &    comm, ierr)
        io = -1
        IF (0 == rank) io = 6
        CALL dbcsr_init_lib(MPI_COMM_WORLD, io)
        CALL dbcsr_set_config(nstacks=NSIZES)

        ALLOCATE(bsize(NBLKS), rdist(NBLKS), cdist(NBLKS))
        DO i = 1, NBLKS
          bsize(i) = SIZES(1 + MOD(4 * i, NSIZES))
          ! the first half of the block rows on one process row
          IF (i <= NBLKS / 2) THEN
            rdist(i) = 0
          ELSE
            rdist(i) = MOD(i, dims(1))
          END IF
          cdist(i) = MOD(i * i, dims(2))
        END DO
        CALL dbcsr_distribution_new(dist, group=comm, row_dist=rdist,   &
     &    col_dist=cdist, reuse_arrays=.TRUE.)
        CALL setup(a, "A", 1)
        CALL setup(b, "B", 2)

        nfail = 0
        DO r = 1, NREPEAT
          DO ta = 1, 2
            DO mode = 0, 2
              CALL product(TRANS(ta), mode, cs(1, mode), cs(2, mode),   &
     &          nb(mode))
            END DO
            ! filtering takes effect, else the comparison proves nothing
            IF (nb(0) <= nb(1)) nfail = nfail + 1
            ! norms too large keep blocks, too small drop them
            IF (nb(1) /= nb(2)                                          &
     &     .OR. TOLERANCE * ABS(cs(1, 1)) < ABS(cs(1, 2) - cs(1, 1))    &
     &     .OR. TOLERANCE * ABS(cs(2, 1)) < ABS(cs(2, 2) - cs(2, 1)))   &
     &      nfail = nfail + 1
            IF (0 == rank) WRITE(*, "(A,I0,3A,3(1X,I0),2ES24.15)")      &
     &        "G2G ", r, " ", TRANS(ta), "N:", nb(0), nb(1), nb(2),     &
     &        cs(1, 1), cs(1, 2)
          END DO
        END DO
        CALL dbcsr_set_config(use_acc_g2g=.FALSE.)
        CALL dbcsr_print_statistics(.TRUE.)

        CALL dbcsr_release(a)
        CALL dbcsr_release(b)
        CALL dbcsr_distribution_release(dist)
        DEALLOCATE(bsize)
        CALL dbcsr_finalize_lib()
        CALL MPI_COMM_FREE(comm, ierr)
        IF (0 == rank .AND. 0 /= nfail) THEN
          WRITE(*, "(A,I0,A)") "ERROR: ", nfail,                        &
     &      " G2G products differ from the host algorithm"
        END IF
        CALL MPI_FINALIZE(ierr)
        IF (0 /= nfail) STOP 1

      CONTAINS

        ! C = op(A) * B unfiltered (0), on the host (1), or by G2G (2)
        SUBROUTINE product(transa, mode, csum, cpos, nblocks)
          CHARACTER(LEN=1), INTENT(IN) :: transa
          INTEGER, INTENT(IN) :: mode
          REAL(DP), INTENT(OUT) :: csum, cpos
          INTEGER, INTENT(OUT) :: nblocks
          TYPE(dbcsr_type) :: c
          INTEGER :: ierr
          CALL dbcsr_set_config(use_acc_g2g=(2 == mode))
          CALL dbcsr_create(c, "C", dist, dbcsr_type_no_symmetry,       &
     &      bsize, bsize, data_type=dbcsr_type_real_8)
          IF (0 == mode) THEN
            CALL dbcsr_multiply(transa, 'N', 1.0_DP, a, b, 0.0_DP, c)
          ELSE
            CALL dbcsr_multiply(transa, 'N', 1.0_DP, a, b, 0.0_DP, c,   &
     &        filter_eps=EPS)
          END IF
          csum = dbcsr_checksum(c)
          cpos = dbcsr_checksum(c, pos=.TRUE.)
          nblocks = dbcsr_get_num_blocks(c)
          CALL MPI_ALLREDUCE(MPI_IN_PLACE, nblocks, 1, MPI_INTEGER,     &
     &      MPI_SUM, MPI_COMM_WORLD, ierr)
          CALL dbcsr_release(c)
        END SUBROUTINE

        ! about 40% of the blocks, scaled from 1 down to 1E-16
        SUBROUTINE setup(matrix, name, seed)
          TYPE(dbcsr_type), INTENT(INOUT) :: matrix
          CHARACTER(LEN=*), INTENT(IN) :: name
          INTEGER, INTENT(IN) :: seed
          REAL(DP), ALLOCATABLE :: blk(:, :)
          REAL(DP) :: s
          INTEGER :: i, j, x, y, h, owner
          CALL dbcsr_create(matrix, name, dist, dbcsr_type_no_symmetry, &
     &      bsize, bsize, data_type=dbcsr_type_real_8)
          DO j = 1, NBLKS
            DO i = 1, NBLKS
              h = MOD(37 * i + 101 * j + 53 * seed, 97)
              IF (h < 40) THEN
                CALL dbcsr_get_stored_coordinates(matrix, i, j, owner)
                IF (owner == rank) THEN
                  ALLOCATE(blk(bsize(i), bsize(j)))
                  s = 10.0_DP ** (-2 * MOD(h, 9))
                  DO y = 1, bsize(j)
                    DO x = 1, bsize(i)
                      blk(x, y) = s * REAL(MOD(7 * x + 3 * y + h, 11)   &
     &                  - 5, DP)
                    END DO
                  END DO
                  CALL dbcsr_put_block(matrix, i, j, blk)
                  DEALLOCATE(blk)
                END IF
              END IF
            END DO
          END DO
          CALL dbcsr_finalize(matrix)
        END SUBROUTINE

      END PROGRAM
