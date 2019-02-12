#! /usr/bin/perl
use File::Compare;
use File::Basename;


$ARGV[0] ||  die "*PATH matrix 1 not found*\n";
$ARGV[1] ||  die "*PATH matrix 2 not found*\n";


$PATH1 = $ARGV[0];
$PATH2 = $ARGV[1];

$TOLERANCE=$ARGV[2];

unless (-f $PATH1) { die "File 1 Doesn't Exist!\n"; }
unless (-f $PATH2) { die "File 2 Doesn't Exist!\n"; }


if ("$PATH1" eq "$PATH2") {
    die "Comparing the same files!\nExiting";
}else{
	
	open(FILE1,$ARGV[0]);
	open(FILE2,$ARGV[1]);
	$count=0;
	$errorSum = 0;
	$max = 0;
	$r=-1;
	$c=-1;
	while($line1=<FILE1>){
	$c=0;
	$r=$r+1;
		$line2=<FILE2>;
		@valuesLine1 = split(/ /,$line1);
		@valuesLine2 = split(/ /,$line2);
		$size = @valuesLine1;
		for($i=0;$i<$size;$i++){
		$c=$c+1;
			if($valuesLine1[$i]!=$valuesLine2[$i]){
				$errorSum+=abs($valuesLine1[$i]-$valuesLine2[$i]);
				$count++;
				if(abs($valuesLine1[$i]-$valuesLine2[$i])>$max){
					$max = abs($valuesLine1[$i]-$valuesLine2[$i]);
				}
			}
		}
	}
	if($max > $TOLERANCE){
		$avgError = $errorSum/$count;
		print "$count:$avgError:$max";
	}else{
		print "OK";
	}
	close(FILE1);
	close(FILE2);

print "\n";
}
	

