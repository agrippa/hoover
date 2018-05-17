/* For license: see LICENSE.txt file at top-level */

#include "shmem_rw_lock.h"
#include <stdio.h>

static void check_state(const long val, const int expected_nreaders,
        const int expected_has_writer) {
    assert(nreaders(val) == expected_nreaders);
    assert(has_writer_request(val) == expected_has_writer);
}

int main(int argc, char **argv) {
    long val = 0;
    check_state(val, 0, 0);

    long one_reader = add_reader(val);
    check_state(one_reader, 1, 0);

    long two_readers = add_reader(one_reader);
    check_state(two_readers, 2, 0);

    long three_readers = add_reader(two_readers);
    check_state(three_readers, 3, 0);

    long three_readers_one_writer = set_writer(three_readers);
    check_state(three_readers_one_writer, 3, 1);

    long zero_readers_one_writer = set_writer(val);
    check_state(zero_readers_one_writer, 0, 1);

    long two_readers_one_writer = remove_reader(three_readers_one_writer);
    check_state(two_readers_one_writer, 2, 1);

    long two_readers_no_writer = clear_writer(two_readers_one_writer);
    check_state(two_readers_no_writer, 2, 0);

    long zero_readers_no_writer = clear_writer(zero_readers_one_writer);
    check_state(zero_readers_no_writer, 0, 0);

    long one_reader_one_writer = set_writer(add_reader(zero_readers_no_writer));
    check_state(one_reader_one_writer, 1, 1);

    zero_readers_one_writer = remove_reader(one_reader_one_writer);
    check_state(zero_readers_one_writer, 0, 1);

    // Invalid: Can't add a reader when there's already a writer
    // two_readers_one_writer = add_reader(one_reader_one_writer);
    // check_state(two_readers_one_writer, 2, 1);

    long one_reader_no_writer = remove_reader(two_readers_no_writer);
    check_state(one_reader_no_writer, 1, 0);

    printf("Passed!\n");

    return 0;
}
