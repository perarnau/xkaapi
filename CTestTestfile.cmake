# CMake generated Testfile for 
# Source directory: /Users/thierry/Developpment/KAAPI/kaapi/git/xkaapi/xkaapi
# Build directory: /Users/thierry/Developpment/KAAPI/kaapi/git/xkaapi/xkaapi
# 
# This file replicates the SUBDIRS() and ADD_TEST() commands from the source
# tree CMakeLists.txt file, skipping any SUBDIRS() or ADD_TEST() commands
# that are excluded by CMake control structures, i.e. IF() commands.
ADD_TEST(create_process "/Users/thierry/Developpment/KAAPI/kaapi/git/xkaapi/xkaapi/tests/test_create1" "1000")
ADD_TEST(create_system "/Users/thierry/Developpment/KAAPI/kaapi/git/xkaapi/xkaapi/tests/test_create2" "100")
ADD_TEST(cond2_process "/Users/thierry/Developpment/KAAPI/kaapi/git/xkaapi/xkaapi/tests/test_cond2" "1")
ADD_TEST(cond2_system "/Users/thierry/Developpment/KAAPI/kaapi/git/xkaapi/xkaapi/tests/test_cond2" "2")
ADD_TEST(cond3_process "/Users/thierry/Developpment/KAAPI/kaapi/git/xkaapi/xkaapi/tests/test_cond3" "1")
ADD_TEST(cond3_system "/Users/thierry/Developpment/KAAPI/kaapi/git/xkaapi/xkaapi/tests/test_cond3" "2")
ADD_TEST(cond4_process "/Users/thierry/Developpment/KAAPI/kaapi/git/xkaapi/xkaapi/tests/test_cond4" "1")
ADD_TEST(cond4_system "/Users/thierry/Developpment/KAAPI/kaapi/git/xkaapi/xkaapi/tests/test_cond4" "2")
ADD_TEST(keys_process "/Users/thierry/Developpment/KAAPI/kaapi/git/xkaapi/xkaapi/tests/test_key" "1")
ADD_TEST(keys_system "/Users/thierry/Developpment/KAAPI/kaapi/git/xkaapi/xkaapi/tests/test_key" "2")
ADD_TEST(once_process "/Users/thierry/Developpment/KAAPI/kaapi/git/xkaapi/xkaapi/tests/test_once" "1")
ADD_TEST(once_system "/Users/thierry/Developpment/KAAPI/kaapi/git/xkaapi/xkaapi/tests/test_once" "2")
ADD_TEST(mutex1 "/Users/thierry/Developpment/KAAPI/kaapi/git/xkaapi/xkaapi/tests/test_mutex1" "200")
ADD_TEST(mutex2 "/Users/thierry/Developpment/KAAPI/kaapi/git/xkaapi/xkaapi/tests/test_mutex2" "200")
ADD_TEST(mutex3 "/Users/thierry/Developpment/KAAPI/kaapi/git/xkaapi/xkaapi/tests/test_mutex3" "200")
ADD_TEST(mutex4_process "/Users/thierry/Developpment/KAAPI/kaapi/git/xkaapi/xkaapi/tests/test_mutex4" "1")
ADD_TEST(mutex4_system "/Users/thierry/Developpment/KAAPI/kaapi/git/xkaapi/xkaapi/tests/test_mutex4" "2")
ADD_TEST(trylock_process "/Users/thierry/Developpment/KAAPI/kaapi/git/xkaapi/xkaapi/tests/test_mutex_trylock" "1")
ADD_TEST(trylock_system "/Users/thierry/Developpment/KAAPI/kaapi/git/xkaapi/xkaapi/tests/test_mutex_trylock" "2")
ADD_TEST(prodcons_process "/Users/thierry/Developpment/KAAPI/kaapi/git/xkaapi/xkaapi/tests/prod-cons" "1")
ADD_TEST(prodcons_system "/Users/thierry/Developpment/KAAPI/kaapi/git/xkaapi/xkaapi/tests/prod-cons" "2")
SUBDIRS(src)
SUBDIRS(tests)
SUBDIRS(doc)
