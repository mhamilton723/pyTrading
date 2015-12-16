__author__ = 'Mark'
import optparse
import os

def main():
	p = optparse.OptionParser()
	p.add_option('--start', default=0)
	p.add_option('--end', default=0)
	ops, args = p.parse_args()

	#ops.start = int(ops.start)
	#ops.end = int(ops.end)
	
	if not ops.start or not ops.end:
		raise ValueError("Use start and end arguments")
	
	print(ops.start,ops.end)
	for job_num in range(int(ops.start),int(ops.end)+1):
		command = 'qdel ' + str(job_num)
		os.system(command)

if __name__ == '__main__':
	main()