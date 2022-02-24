from sys import argv
import decoding_flashlight
import bcrypt

test_case_dirs = ['test1.mov', 'test-2.mov', 'test-1.mov']
encrypted_texts = [b'$2b$10$5S3tcCUUSKTHW/SnbhRRb.XzAF5Odqd9JXx7QLmLF2Y5bskfYAPq2',
                   b'$2b$10$5S3tcCUUSKTHW/SnbhRRb.S5Xh07Eed6NgnPTuI1XTFU4qGV0ota2',
                   b'$2b$10$5S3tcCUUSKTHW/SnbhRRb.PYE4tMp0szACPNKbh97vcaP0Qbdkf3y']

def run_test(test_case, salt=b'$2b$10$5S3tcCUUSKTHW/SnbhRRb.'):
    student_answer = decoding_flashlight.run('inputs/'+test_case_dirs[test_case])
    correct_answer = encrypted_texts[test_case]
    if bcrypt.hashpw(bytes(student_answer, 'utf-8'), salt) == correct_answer:
        print("Test "+str(test_case+1)+': Passed')
    else:
        print("Test "+str(test_case+1)+': Failed')

def main():
    test_case = argv[1]
    if test_case == 'all_tests':
        for i in range(len(test_case_dirs)):
            run_test(i)
    else:
        run_test(int(test_case)-1)

if __name__ == "__main__":
    main()