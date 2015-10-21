
<!DOCTYPE HTML PUBLIC "-//W3C//DTD HTML 4.0 Transitional//EN">
<!-- saved from url=(0053)CppCodingStandard.html -->
<HTML><HEAD><TITLE>C Coding Standard</TITLE>
<META http-equiv=Content-Type content="text/html; charset=windows-1252">
<META content="Microsoft FrontPage 5.0" name=GENERATOR></HEAD>
<BODY bgColor=#ffffff>
<CENTER>
<H1><i>C</i><span style="font-weight: 400"> Coding Standard
</span> </H1>
<H4>Adapted from <a href="http://www.possibility.com/Cpp/CppCodingStandard.html">http://www.possibility.com/Cpp/CppCodingStandard.html</a> and NetBSD's style guidelines</CENTER>
<P>
For the C++ coding standards click <a href="CppCodingStandard.html">here</a>
</P>
<P>
<hr>
<H1>Contents</H1>
<OL>
  <LI><A 
  href="#names"><B>Names</B> 
  </A>
  <UL>
    <LI><i>(important recommendations below)</i>
    <LI><A 
    href="#units">Include 
    Units in Names </A>
    <LI><a href="#classnames">Structure Names </a>
    <LI><A href="#fext">C 
	File Extensions </A></LI>
    <LI><i>(other suggestions below)</i>
    <LI><A 
	href="#descriptive">Make 
	Names Fit </A>
	
    <LI><A 
	href="#stacknames">Variable 
	Names on the Stack </A>
    <LI><A 
	href="#pnames">Pointer 
	Variables </A>
    <LI><A 
	href="#gconstants">Global 
	Constants </A>
    <LI><A 
	href="#enames">Enum 
	Names </A>
    <LI><A 
	href="#mnames">#define 
	and Macro Names </A>
  </UL><A name=docidx></A>
   <LI><A 
  href="#formatting"><B>Formatting</B>
      </A>
  <UL>
    <LI><i>(important recommendations below)</i>
    <LI><A 
    href="#brace">Brace 
    <I>{}</I> Policy </A>
    <LI><A 
    href="#parens">Parens 
    <I>()</I> with Key Words and Functions Policy </A>
    <LI><A 
    href="#linelen">A Line 
    Should Not Exceed 78 Characters </A>
    <LI><A 
    href="#ifthen"><I>If 
    Then Else</I> Formatting </A>
    <LI><A 
    href="#switch"><I>switch</I> 
    Formatting </A>
    <LI><A href="#goto">Use 
    of <I>goto,continue,break</I> and <I>?:</I> </A>
    <LI><i>(other suggestions below)</i>
    <LI><A href="#one">One 
    Statement Per Line </A>
    </UL>
    <LI><A href="#documentation"><B>Documentation</B>
	</A> 
  <UL>
    <LI><i>(important recommendations below)</i>
    <LI><A 
    href="#cstas">Comments 
    Should Tell a Story </A>
    <LI><A 
    href="#cdd">Document 
    Decisions </A>
    <LI><A href="#cuh">Use 
    Headers </A>
    <LI><A href="#mge">Make 
    Gotchas Explicit </A>
    <LI><A href="#cdef">Commenting function declarations </A>
    <LI><i>(other suggestions below)</i>
    <LI><A 
    href="#idoc">Include 
    Statement Documentation </A>
    </UL>
  <LI><A 
  href="#complexity"><B>Complexity Management</B> </A>
    <UL>
    <LI><A 
    href="#layering">Layering 
    </A>
    </UL>
 <LI><A 
  href="#misc"><B>Miscellaneous</B> 
     </A>
     <UL>
    <LI><i>(important recommendations below)</i>
    <LI><A 
    href="#guards">Use 
    Header File Guards </A>
    <LI><A 
    href="#callc">Mixing C 
    and C++ </A>
    <LI><i>(other suggestions below)</i>
    <LI><A 
    href="#initvar">Initialize 
    all Variables </A>
    <LI><A href="#const">Be 
    Const Correct </A>
    <LI><A 
    href="#shortmethods">Short Functions</A></LI>
    <LI><A 
    href="#nomagic">No 
    Magic Numbers 
    <LI><A 
    href="#errorret">Error 
    Return Check Policy 
    <LI><A 
    href="#useenums">To Use 
    Enums or Not to Use Enums </A>
    <LI><A 
    href="#macros">Macros 
    </A>
    <LI><A 
    href="#nztest">Do Not 
    Default If Test to Non-Zero </A>
    <LI><A 
    href="#eassign">Usually 
    Avoid Embedded Assignments</A><LI><A 
    href="#if0">Commenting 
    Out Large Code Blocks 
    <LI><A 
    href="#ifdef">Use #if 
    Not #ifdef 
    <LI><A 
    href="#misc">Miscellaneous 
    </A>
    <LI><A href="#nodef">No 
    Data Definitions in Header Files </A>
    </UL>
  </OL>

<HR>
<A name=names></A>
<H1>Names </H1><A name=descriptive></A>
  <HR>
<H2>Make Names Fit </H2>Names are the heart of programming. In the past people 
believed knowing someone's true name gave them magical power over that person. 
If you can think up the true name for something, you give yourself and the 
people coming after power over the code. Don't laugh! 
<P>A name is the result of a long deep thought process about the ecology it 
lives in. Only a programmer who understands the system as a whole can create a 
name that "fits" with the system. If the name is appropriate everything fits 
together naturally, relationships are clear, meaning is derivable, and reasoning 
from common human expectations works as expected. 
<P>If you find all your names could be Thing and DoIt then you should probably 
revisit your design.
  <HR>
<H2>Function Names </H2>
<UL>
  <LI>Usually every  function performs an action, so the name should 
  make clear what it does: check_for_errors() instead of error_check(), 
  dump_data_to_file() instead of data_file(). This will also make functions and data 
  objects more distinguishable. 
  <P>Structs are often nouns. By making function names verbs and following other 
  naming conventions programs can be read more naturally. 
  <P></P>
  <LI>Suffixes are sometimes useful: 
  <UL>
    <LI><i>max</i> - to mean the maximum value something can have. 
    <LI><i>cnt</i> - the current count of a running count variable. 
    <LI><i>key</i> - key value. </LI></UL>
  <P>For example: retry_max to mean the maximum number of retries, retry_cnt to 
  mean the current retry count. 
  <P></P>
  <LI>Prefixes are sometimes useful: 
  <UL>
    <LI><i>is</i> - to ask a question about something. Whenever someone sees 
    <I>Is</I> they will know it's a question. 
    <LI><i>get</i> - get a value. 
    <LI><i>set</i> - set a value. </LI></UL>
  <P>For example: is_hit_retry_limit. 
  <P></P></LI></UL><A name=units></A>
  <HR>
<H2>Include Units in Names </H2>If a variable represents time, weight, or some 
other unit then include the unit in the name so developers can more easily spot 
problems. For example: <PRE>uint32 timeout_msecs;
uint32 my_weight_lbs;

</PRE>
<P>
<HR>
<A name=classnames></A>
<H2>Structure Names </H2>
<UL>
  <LI>Use underbars ('_') to separate name components </LI> 
  <LI>When declaring variables in structures, declare them organized by use in
  a manner to attempt to minimize memory wastage because of compiler alignment
  issues, then by size, and then by alphabetical order. E.g, don't use
  ``int a; char *b; int c; char *d''; use ``int a; int b; char *c; char *d''. Each variable gets its own type and line, although an exception can be 
  made
  when declaring bitfields (to clarify that it's part of the one bitfield).
  Note that the use of bitfields in general is discouraged.
  Major structures should be declared at the top of the file in which they
 are used, or in separate header files, if they are used in multiple
  source files. Use of the structures should be by separate declarations
  and should be &quot;extern&quot; if they are declared in a header file.
  It may be useful to use a meaningful prefix for each member name. E.g, for ``struct softc'' the prefix could be ``sc_''.</LI></UL>
<H3>Example </H3><PRE>   
struct foo {
	struct foo *next;	/* List of active foo */
	struct mumble amumble;	/* Comment for mumble */
	int bar;
	unsigned int baz:1,	/* Bitfield; line up entries if desired */
		     fuz:5,
		     zap:2;
	uint8_t flag;
};
struct foo *foohead;		/* Head of global foo list */

</PRE>
<P>
  <HR>
<A name=stacknames></A>
<H2>Variable Names on the Stack </H2>
<UL>
  <LI>use all lower case letters 
  <LI>use '_' as the word separator. </LI></UL>
<H3>Justification </H3>
<UL>
  <LI>With this approach the scope of the variable is clear in the code. 
  <LI>Now all variables look different and are identifiable in the code. 
</LI></UL>
<H3>Example </H3><PRE>   
   int handle_error (int error_number) {
      int            error= OsErr();
      Time           time_of_error;
      ErrorProcessor error_processor;
   }
</PRE>
<P>
<HR>
<A name=pnames></A>
<H2>Pointer Variables </H2>
<UL>
  <LI>place the <I>*</I> close to the variable name not pointer type
</LI></UL>

<H3>Example </H3><PRE>  char *name= NULL;

  char *name, address; 
</PRE>
<P>
<HR>
<H2>Global Variables </H2>
<UL>
  <LI>Global variables should be prepended with a 'g_'. </LI>
  <LI>Global variables should be avoided whenever possible. </LI>
</UL>
  
  <H3>Justification </H3>
<UL>
  <LI>It's important to know the scope of a variable. </LI></UL>
<H3>Example </H3><PRE>    Logger  g_log;
    Logger* g_plog;
</PRE>
<P>
<HR>
<A name=gconstants></A>
<H2>Global Constants </H2>
<UL>
  <LI>Global constants should be all caps with '_' separators. </LI></UL>
<H3>Justification </H3>It's tradition for global constants to named this way. 
You must be careful to not conflict with other global <I>#define</I>s and enum 
labels. 
<H3>Example </H3><PRE>    const int A_GLOBAL_CONSTANT= 5;</PRE>
<A name=snames></A>
<P>
<HR>
<A name=mnames></A>
<H2>#define and Macro Names </H2>
<UL>
  <LI>Put #defines and macros in all upper using '_' separators. 
  Macros are capitalized, parenthesized, and should avoid side-effects.
  Spacing before and after the macro name may be any whitespace, though
  use of TABs should be consistent through a file.
  If they are an inline expansion of a function, the function is defined
  all in lowercase, the macro has the same name all in uppercase.
  If the macro is an expression, wrap the expression in parenthesis.
  If the macro is more than a single statement, use ``do { ... } while (0)'',
  so that a trailing semicolon works.  Right-justify the backslashes; it
  makes it easier to read.
 </LI></UL>
<H3>Justification </H3>This makes it very clear that the value is not alterable 
and in the case of macros, makes it clear that you are using a construct that 
requires care. 
<P>Some subtle errors can occur when macro names and enum labels use the same 
name. 
<H3>Example </H3><PRE>#define MAX(a,b) blah
#define IS_ERR(err) blah
#define	MACRO(v, w, x, y)						\
do {									\
	v = (x) + (y);							\
	w = (y) + 2;							\
} while (0)
</PRE>
<A name=cnames></A>
<P>
<HR>
<A name=enames></A>
<H2>Enum Names </H2>
<H3>Labels All Upper Case with '_' Word Separators </H3>This is the standard 
rule for enum labels. No comma on the last element.<H4>Example </H4><PRE>   enum PinStateType {
      PIN_OFF,
      PIN_ON
   };
</PRE>
<H3>Make a Label for an Error State </H3>It's often useful to be able to say an 
enum is not in any of its <I>valid</I> states. Make a label for an uninitialized 
or error state. Make it the first label if possible. 
<H4>Example </H4><PRE>enum { STATE_ERR,  STATE_OPEN, STATE_RUNNING, STATE_DYING};</PRE>
<A name=req></A>
<P>
<HR>
 <A name=formatting></A>
 <H1>Formatting </H1>
 <HR>
 <A name=brace></A>
<H2>Brace Placement </H2>Of the three major brace placement strategies one is 
recommended: 
<UL>
   <PRE>   if (condition) {      while (condition) {
      ...                   ...
   }                     }
</PRE></LI></UL>
<HR>
<H2>When Braces are Needed </H2>All if, while and do statements must either have 
braces or be on a single line. 
<P>
<H3>Always Uses Braces Form </H3>All if, while and do statements require braces 
even if there is only a single statement within the braces. For example: <PRE>if (1 == somevalue) {
   somevalue = 2;
}
</PRE>
<H4>Justification </H4>It ensures that when someone adds a line of code later 
there are already braces and they don't forget. It provides a more consistent 
look. This doesn't affect execution speed. It's easy to do. 
<H3>Do Not Use One Line Form</H3>The following form should not be used
<PRE>if (1 == somevalue) somevalue = 2;
</PRE>
<HR>
<H2>Add Comments to Closing Braces </H2>Adding a comment to closing braces can 
help when you are reading code because you don't have to find the begin brace to 
know what is going on. <PRE>while(1) {
   if (valid) {
  
   } /* if valid */
   else {
   } /* not valid */

} /* end forever */
</PRE>
<HR>
<H2>Consider Screen Size Limits </H2>Some people like blocks to fit within a 
common screen size so scrolling is not necessary when reading code. <br>
<P>
<HR>
<A name=parens></A>
<H2>Parens <I>()</I> with Key Words and Functions Policy </H2>
<UL>
  <LI>Do not put parens next to keywords. Put a space between. 
  <LI>Do put parens next to function names. 
  <LI>Do not use parens in return statements when it's not necessary. </LI></UL>
<H3>Justification </H3>
<UL>
  <LI>Keywords are not functions. By putting parens next to keywords keywords 
  and function names are made to look alike. </LI></UL>
<H3>Example </H3><PRE>    if (condition) {
    }

    while (condition) {
    }

    strcpy(s, s1);

    return 1;</PRE>
<HR>
<A name=linelen></A>
<H2>A Line Should Not Exceed 78 Characters </H2>
<UL>
  <LI>Lines should not exceed 78 characters. </LI></UL>
<H2>Justification </H2>
<UL>
  <LI>Even though with big monitors we stretch windows wide our printers can 
  only print so wide. And we still need to print code. 
  <LI>The wider the window the fewer windows we can have on a screen. More 
  windows is better than wider windows. 
  <LI>We even view and print diff output correctly on all terminals and 
  printers. </LI></UL>
<P>
<HR>
<A name=ifthen></A>
<H2><I>If Then Else</I> Formatting </H2>
<H3>Layout </H3>It's up to the programmer. Different bracing styles will yield 
slightly different looks. One common approach is: <PRE>   if (condition) {
   } else if (condition) {
   } else {
   }
</PRE>If you have <I>else if</I> statements then it is usually a good idea to 
always have an else block for finding unhandled cases. Maybe put a log message 
in the else even if there is no corrective action taken. 
<P>
<H3>Condition Format </H3>Always put the constant on the left hand side of an 
equality/inequality comparison. For example: 
<P>if ( 6 == errorNum ) ... 
<P>One reason is that if you leave out one of the = signs, the compiler will 
find the error for you. A second reason is that it puts the value you are 
looking for right up front where you can find it instead of buried at the end of 
your expression. It takes a little time to get used to this format, but then it 
really gets useful. 
<P>
<P>
<HR>
<A name=switch></A>
<H2><I>switch</I> Formatting </H2>
<UL>
  <LI>Falling through a case statement into the next case statement shall be 
  permitted as long as a comment is included. 
  <LI>The <I>default</I> case should always be present and trigger an error if 
  it should not be reached, yet is reached. 
  <LI>If you need to create variables put all the code in a block. </LI></UL>
<H3>Example </H3><PRE>   switch (...)
   {
      case 1:
         ...
      /* comments */

      case 2:
      {        
         int v;
         ...
      }
      break;

      default:
   }
</PRE>
<P>
<HR>
<A name=goto></A>
<H2>Use of <I>goto,continue,break</I> and <I>?:</I> </H2>
<H3>Goto </H3>Goto statements should be used sparingly, as in any 
well-structured code. The goto debates are boring so we won't go into them here. 
The main place where they can be usefully employed is to break out of several 
levels of switch, for, and while nesting, although the need to do such a thing 
may indicate that the inner constructs should be broken out into a separate 
function, with a success/failure return code. 
<P><PRE><TT>
   for (...) {
      while (...) {
      ...
         if (disaster) {
            goto error;</TT></PRE>
<PRE>         } <TT>
      }
   }
   ...
error:
   clean up the mess 
</TT></PRE>
<P>When a goto is necessary the accompanying label should be alone on a line and 
to the left of the code that follows. The goto should be commented (possibly in 
the block header) as to its utility and purpose. <A name=contbreak></A>
<H3>Continue and Break </H3>Continue and break are really disguised gotos so 
they are covered here. 
<P>Continue and break like goto should be used sparingly as they are magic in 
code. With a simple spell the reader is beamed to god knows where for some 
usually undocumented reason. 
<P>The two main problems with continue are: 
<UL>
  <LI>It may bypass the test condition 
  <LI>It may bypass the increment/decrement expression </LI></UL>
<P>Consider the following example where both problems occur: <PRE>while (TRUE) {
   ...
   /* A lot of code */
   ...
   if (/* some condition */) {
      continue;
   }
   ...
   /* A lot of code */
   ...
   if ( i++ &gt; STOP_VALUE) break;
}
</PRE>Note: "A lot of code" is necessary in order that the problem cannot be 
caught easily by the programmer. 
<P>From the above example, a further rule may be given: Mixing continue with 
break in the same loop is a sure way to disaster. 
<P>
<H3>?: </H3>The trouble is people usually try and stuff too much code in between 
the <I>?</I> and <I>:</I>. Here are a couple of clarity rules to follow: 
<UL>
  <LI>Put the condition in parens so as to set it off from other code 
  <LI>If possible, the actions for the test should be simple functions. 
  <LI>Put the action for the then and else statement on a separate line unless 
  it can be clearly put on one line. </LI></UL>
<H3>Example </H3><PRE>   (condition) ? funct1() : func2();

   or

   (condition)
      ? long statement
      : another long statement;</PRE>
<A name=aligndecls></A>
<P>
<hr>
  <A name=one></A>
<H2>One Statement Per Line </H2>There should be only one statement per line 
unless the statements are very closely related. 
<P>The reasons are: 
<OL>
  <LI>The code is easier to read. Use some white space too. Nothing better than 
  to read code that is one line after another with no white space or comments. 
  </LI></OL>
<H3>One Variable Per Line </H3>Related to this is always define one variable per 
line: <PRE><B>Not:</B>
char **a, *x;

<B>Do</B>:
char **a = 0;  /* add doc */
char  *x = 0;  /* add doc */
</PRE>The reasons are: 
<OL>
  <LI>Documentation can be added for the variable on the line. 
  <LI>It's clear that the variables are initialized. 
  <LI>Declarations are clear which reduces the probablity of declaring a pointer 
  when you meant to declare just a char. </LI></OL>
<HR>
<A name=useenums></A>
<H2>To Use Enums or Not to Use Enums </H2>C allows constant variables, which 
should deprecate the use of enums as constants. Unfortunately, in most compilers 
constants take space. Some compilers will remove constants, but not all. 
Constants taking space precludes them from being used in tight memory 
environments like embedded systems. Workstation users should use constants and 
ignore the rest of this discussion. 
<P>In general enums are preferred to <I>#define</I> as enums are understood by 
the debugger. 
<P>Be aware enums are not of a guaranteed size. So if you have a type that can 
take a known range of values and it is transported in a message you can't use an 
enum as the type. Use the correct integer size and use constants or 
<I>#define</I>. Casting between integers and enums is very error prone as you 
could cast a value not in the enum. 
  <PRE>&nbsp;</PRE>
<P>
<HR>
<A name=guards></A>
<H2>Use Header File Guards </H2>Include files should protect against multiple 
  inclusion through the use of macros that "guard" the files. Note
  that for C++ compatibility and interoperatibility reasons,
  do <b>not</b> use underscores '_' as the first or last character
  of a header guard (see below)
  
  <PRE>#ifndef sys_socket_h
  #define sys_socket_h  /* NOT _sys_socket_h_ */
  #endif 
  </PRE>
  
  <P></P>
<P>
<HR>
<A name=macros></A>
<H1>Macros </H1>
  <H2>Don't Turn C into Pascal </H2>Don't change syntax via macro substitution. 
  It makes the program unintelligible to all but the perpetrator. 
  <H2>Replace Macros with Inline Functions </H2>In C macros are not needed for 
  code efficiency. Use inlines. However, macros for
  small functions are ok.
<H3>Example </H3><PRE>#define  MAX(x,y)	(((x) &gt; (y) ? (x) : (y))	// Get the maximum
</PRE>
<P>The macro above can be replaced for integers with the following inline 
function with no loss of efficiency: <PRE>   inline int 
   max(int x, int y) {
      return (x &gt; y ? x : y);
   }
</PRE>
<H2>Be Careful of Side Effects </H2>Macros should be used with caution because 
of the potential for error when invoked with an expression that has side 
effects. 
<H3>Example </H3><PRE>   MAX(f(x),z++);
</PRE>
<H2>Always Wrap the Expression in Parenthesis </H2>When putting expressions in 
macros always wrap the expression in parenthesis to avoid potential communitive 
operation abiguity. 
<H3>Example </H3><PRE>#define ADD(x,y) x + y

must be written as 

#define ADD(x,y) ((x) + (y))
</PRE>
<H2>Make Macro Names Unique </H2>Like global variables macros can conflict with 
macros from other packages. 
<OL>
  <LI>Prepend macro names with package names. 
  <LI>Avoid simple and common names like MAX and MIN. </LI></OL>
<P>
<HR>
<A name=initvar></A>
<H1>Initialize all Variables </H1>
<UL>
  <LI>You shall always initialize variables. Always. Every time. gcc with the flag -W may catch operations on uninitialized variables, but it may also not.</LI></UL>
<H2>Justification </H2>
<UL>
  <LI>More problems than you can believe are eventually traced back to a pointer 
  or variable left uninitialized. </LI></UL>
<A name=init></A>
<P>
<HR>
<A name=shortmethods></A>
<H1>Short Functions </H1>
<UL>
  <LI>Functions should limit themselves to a single page of code. </LI></UL>
<H3>Justification </H3>
<UL>
  <LI>The idea is that the each method represents a technique for achieving a 
  single objective. 
  <LI>Most arguments of inefficiency turn out to be false in the long run. 
  <LI>True function calls are slower than not, but there needs to a thought out 
  decision (see premature optimization). </LI></UL>
<P>
<HR>
<A name=docnull></A>
<H1>Document Null Statements </H1>Always document a null body for a for or while 
statement so that it is clear that the null body is intentional and not missing 
code. <PRE><TT>
   while (*dest++ = *src++) </TT></PRE>
  <PRE><TT>  {
      ;       </TT></PRE>
  <PRE>   }<TT>  
</TT></PRE>
<P>
<HR>
<A name=nztest></A>
<H1>Do Not Default If Test to Non-Zero </H1>Do not default the test for 
non-zero, i.e. <PRE><TT>
   if (FAIL != f()) 
</TT></PRE>is better than <PRE><TT>
   if (f()) 
</TT></PRE>even though FAIL may have the value 0 which C considers to be false. 
An explicit test will help you out later when somebody decides that a failure 
return should be -1 instead of 0. Explicit comparison should be used even if the 
comparison value will never change; e.g., <B>if (!(bufsize % sizeof(int)))</B> 
should be written instead as <B>if ((bufsize % sizeof(int)) == 0)</B> to reflect 
the numeric (not boolean) nature of the test. A frequent trouble spot is using 
strcmp to test for string equality, where the result should <EM>never</EM> 
<EM>ever</EM> be defaulted. The preferred approach is to define a macro 
<EM>STREQ</EM>. 
<P><PRE><TT>
   #define STREQ(a, b) (strcmp((a), (b)) == 0) 
</TT></PRE>
<P>Or better yet use an inline method: <PRE><TT>
   inline bool
   string_equal(char* a, char* b)
   {
      (strcmp(a, b) == 0) ? return true : return false;
	  Or more compactly:
      return (strcmp(a, b) == 0);
   }
</TT></PRE>
<P>Note, this is just an example, you should really use the standard library 
string type for doing the comparison. 
<P>The non-zero test is often defaulted for predicates and other functions or 
expressions which meet the following restrictions: 
<UL>
  <LI>Returns 0 for false, nothing else. 
  <LI>Is named so that the meaning of (say) a <B>true</B> return is absolutely 
  obvious. Call a predicate is_valid(), not check_valid(). </LI></UL>
<A name=boolean></A>
<P>
<HR>
<A name=eassign></A>
<H1>Usually Avoid Embedded Assignments </H1>There is a time and a place for 
embedded assignment statements. In some constructs there is no better way to 
accomplish the results without making the code bulkier and less readable. 
<P><PRE><TT>
   while (EOF != (c = getchar())) {
      process the character
   }
</TT></PRE>
<P>The ++ and -- operators count as assignment statements. So, for many 
purposes, do functions with side effects. Using embedded assignment statements 
to improve run-time performance is also possible. However, one should consider 
the tradeoff between increased speed and decreased maintainability that results 
when embedded assignments are used in artificial places. For example, <PRE><TT>
   a = b + c;
   d = a + r; 
</TT></PRE>should not be replaced by <PRE><TT>
   d = (a = b + c) + r; 
</TT></PRE>even though the latter may save one cycle. In the long run the time 
difference between the two will decrease as the optimizer gains maturity, while 
the difference in ease of maintenance will increase as the human memory of 
what's going on in the latter piece of code begins to fade. 

<P>

  <HR>
<A name=documentation></A>
<H1>Documentation </H1>
  <HR>
<A name=cstas></A>
<H2>Comments Should Tell a Story </H2>Consider your comments a story describing 
the system. Expect your comments to be extracted by a robot and formed into a 
man page. Class comments are one part of the story, method signature comments 
are another part of the story, method arguments another part, and method 
implementation yet another part. All these parts should weave together and 
inform someone else at another point of time just exactly what you did and why. <hr>
<A name=cdd></A>
<H2>Document Decisions </H2>Comments should document decisions. At every point 
where you had a choice of what to do place a comment describing which choice you 
made and why. Archeologists will find this the most useful information. <A 
name=cuh></A>
 <hr>
<H2>Use Headers </H2>Use a document extraction system like
  <a href="http://www.doxygen.org">Doxygen</a>. 
<P>These headers are structured in such a way as they can be parsed and 
extracted. They are not useless like normal headers. So take time to fill them 
out. If you do it right once no more documentation may be necessary.<P>
 <hr>
<H2>Comment Layout </H2>Each part of the project has a specific comment layout.
<a href="http://www.doxygen.org">Doxygen</a> has the recommended
format for the comment layouts.
 <hr>
<A name=mge></A>
<H2>Make Gotchas Explicit </H2>Explicitly comment variables changed out of the 
normal control flow or other code likely to break during maintenance. Embedded 
keywords are used to point out issues and potential problems. Consider a robot 
will parse your comments looking for keywords, stripping them out, and making a 
report so people can make a special effort where needed. 
<P>

<H3>Gotcha Keywords </H3>
<UL>
<LI><B>@author:</B><BR> specifies the author of the module <P></P>
<LI><B>@version:</B><BR> specifies the version of the module <P></P>
<LI><B>@param:</B><BR> specifies a parameter into a function <P></P>
<LI><B>@return:</B><BR> specifies what a function returns <P></P>
<LI><B>@deprecated:</B><BR> says that a function is not to be used anymore <P></P>
<LI><B>@see:</B><BR> creates a link in the documentation to the 
file/function/variable to consult to get a better understanding on what the current block of code does. 
<LI><B>@todo:</B><BR> what remains to be done<P></P>
<LI><B>@bug:</B><BR> report a bug found in the piece of code<P></P>
</LI></UL>

<P>
<H3>Gotcha Formatting </H3>
<UL>
  <LI>Make the gotcha keyword the first symbol in the comment. 
  <LI>Comments may consist of multiple lines, but the first line should be a 
  self-containing, meaningful summary. 
  <LI>The writer's name and the date of the remark should be part of the 
  comment. This information is in the source repository, but it can take a quite 
  a while to find out when and by whom it was added. Often gotchas stick around 
  longer than they should. Embedding date information allows other programmer to 
  make this decision. Embedding who information lets us know who to ask. 
</LI></UL>
<A name=cdef></A>
<H2>Commenting function declarations</H2>
Functions headers should be in the file where
they are declared. This means that most likely
the functions will have a header in the .h file. However,
functions like main() with no explicit
prototype declaration in the .h file, should have a header
in the .c file.

<A name=cflayout></A>
<P>
<HR>
<A name=idoc></A>
<H2>Include Statement Documentation</H2>Include statements should be documented, 
  telling the user why a particular file was included. 
  <br>
/* <br>
 * Kernel include files come first.<br>
 */<br>
 /* Non-local includes in brackets. */<br>

/*<br>
 * If it's a network program, put the network include files next.<br>
 * Group the includes files by subdirectory.<br>
 */<br>

/*<br>
 * Then there's a blank line, followed by the /usr include files.<br>
 * The /usr include files should be sorted!<br>
 */<P>
<hr>
<A name=layering></A>
<H1>Layering </H1>
<hr>
Layering is the primary technique for reducing complexity in a 
system. A system should be divided into layers. Layers should communicate 
between adjacent layers using well defined interfaces. When a layer uses a 
non-adjacent layer then a layering violation has occurred. 
<P>A layering violation simply means we have dependency between layers that is 
not controlled by a well defined interface. When one of the layers changes code 
could break. We don't want code to break so we want layers to work only with 
other adjacent layers. 
<P>Sometimes we need to jump layers for performance reasons. This is fine, but 
we should know we are doing it and document appropriately. 
<hr>

<A name=misc></A>
<H1>Miscellaneous </H1>
<hr>
<H2>General advice</H2>
This section contains some miscellaneous do's and don'ts. 

<P>
<UL>
  <LI>Don't use floating-point variables where discrete values are needed. Using 
  a float for a loop counter is a great way to shoot yourself in the foot. 
  Always test floating-point numbers as &lt;= or &gt;=, never use an exact 
  comparison (== or !=). 
  <P></P>
  <LI>Compilers have bugs. Common trouble spots include structure assignment and 
  bit fields. You cannot generally predict which bugs a compiler has. You could 
  write a program that avoids all constructs that are known broken on all 
  compilers. You won't be able to write anything useful, you might still 
  encounter bugs, and the compiler might get fixed in the meanwhile. Thus, you 
  should write ``around'' compiler bugs only when you are forced to use a 
  particular buggy compiler. 
  <P></P>
  <LI>Do not rely on automatic beautifiers. The main person who benefits from 
  good program style is the programmer him/herself, and especially in the early 
  design of handwritten algorithms or pseudo-code. Automatic beautifiers can 
  only be applied to complete, syntactically correct programs and hence are not 
  available when the need for attention to white space and indentation is 
  greatest. Programmers can do a better job of making clear the complete visual 
  layout of a function or file, with the normal attention to detail of a careful 
  programmer (in other words, some of the visual layout is dictated by intent 
  rather than syntax and beautifiers cannot read minds). Sloppy programmers 
  should learn to be careful programmers instead of relying on a beautifier to 
  make their code readable. Finally, since beautifiers are non-trivial programs 
  that must parse the source, a sophisticated beautifier is not worth the 
  benefits gained by such a program. Beautifiers are best for gross formatting 
  of machine-generated code. 
  <P></P>
  <LI>Accidental omission of the second ``='' of the logical compare is a 
  problem. The following is confusing and prone to error. <PRE>        if (abool= bbool) { ... }
     </PRE>Does the programmer really mean assignment here? Often yes, but 
  usually no. The solution is to just not do it, an inverse Nike philosophy. 
  Instead use explicit tests and avoid assignment with an implicit test. The 
  recommended form is to do the assignment before doing the test: <PRE><TT>
       abool= bbool;
       if (abool) { ... }
    </TT></PRE>
  <P></P>
  <LI>Modern compilers will put variables in registers automatically. Use the 
  register sparingly to indicate the variables that you think are most critical. 
  In extreme cases, mark the 2-4 most critical values as register and mark the 
  rest as REGISTER. The latter can be #defined to register on those machines 
  with many registers. </LI></UL>
<P>
<HR>
<A name=const></A>
<H2>Be Const Correct </H2>C provides the <I>const</I> key word to allow 
passing as parameters objects that cannot change to indicate when a method 
doesn't modify its object. Using const in all the right places is called "const 
correctness." It's hard at first, but using const really tightens up your coding 
style. Const correctness grows on you. 
<A name=streams></A>
<P>
<HR>
<A name=ifdef></A>
<H2>Use #if Not #ifdef </H2>Use #if MACRO not #ifdef MACRO. Someone might write 
code like: <PRE>#ifdef DEBUG
        temporary_debugger_break();
#endif
</PRE>Someone else might compile the code with turned-of debug info like: <PRE>cc -c lurker.cc -DDEBUG=0
</PRE>Alway use #if, if you have to use the preprocessor. This works fine, and 
does the right thing, even if DEBUG is not defined at all (!) <PRE>#if DEBUG
        temporary_debugger_break();
#endif
</PRE>If you really need to test whether a symbol is defined or not, test it 
with the defined() construct, which allows you to add more things later to the 
conditional without editing text that's already in the program: <PRE>#if !defined(USER_NAME)
 #define USER_NAME "john smith"
#endif
</PRE>
<P>
<HR>
<A name=if0></A>
<H2>Commenting Out Large Code Blocks </H2>Sometimes large blocks of code need to 
be commented out for testing. 
<H3>Using #if 0 </H3>The easiest way to do this is with an #if 0 block: <PRE>   void 
   example()
   {
      great looking code

      #if 0
      lots of code
      #endif
    
      more code
    }
</PRE>
<P>You can't use <B>/**/</B> style comments because comments can't contain 
comments and surely a large block of your code will contain a comment, won't it? 

<P>Don't use #ifdef as someone can unknowingly trigger ifdefs from the compiler 
command line. 
<H3Use Descriptive Macro Names Instead of 0 </H3The problem with <B>#if 
0</B>is that even day later you or anyone else has know idea why this code is 
commented out. Is it because a feature has been dropped? Is it because it was 
buggy? It didn't compile? Can it be added back? It's a mystery. 
<H3>Use Descriptive Macro Names Instead of #if 0 </H3><PRE>#if NOT_YET_IMPLEMENTED  

#if OBSOLETE

#if TEMP_DISABLED 
</PRE>
<H3>Add a Comment to Document Why </H3>Add a short comment explaining why it is 
not implemented, obsolete or temporarily disabled. 
<A name=accessor></A>
  <PRE>&nbsp;</PRE>


<A name=personas></A>
<HR>
<A name=fext>
<H2>File Extensions </H2>In short: Use the <I>.h</I> extension for header 
files and <I>.c </I> for source files. 
<P>
<P>
<HR>
<A name=nodef></A>
<H2>No Data Definitions in Header Files </H2>Do not put data definitions in 
header files. for example: <PRE>/* 
 * aheader.h 
 */
int x = 0;
</PRE>
<P>
<OL>
  <LI>It's bad magic to have space consuming code silently inserted through the 
  innocent use of header files. 
  <LI>It's not common practice to define variables in the header file so it will 
  not occur to developers to look for this when there are problems. 
  <LI>Consider defining the variable once in a .c file and use an extern 
  statement to reference it. 
</OL>
      <P>
<HR>
<A name=callc></A>
<H2>Mixing C and C++ </H2>In order to be backward compatible with dumb linkers 
  C++'s link time type safety is implemented by encoding type information in 
  link symbols, a process called <I>name mangling</I>. This creates a problem 
  when linking to C code as C function names are not mangled. When calling a C 
  function from C++ the function name will be mangled unless you turn it off. 
  Name mangling is turned off with the <I>extern &quot;C&quot;</I> syntax. If you want to 
  create a C function in C++ you must wrap it with the above syntax. If you want 
  to call a C function in a C library from C++ you must wrap in the above 
  syntax. Here are some examples:
<P>
<H3>Calling C Functions from C++ </H3><PRE>extern &quot;C&quot; int strncpy(...);
extern &quot;C&quot; int my_great_function();
extern &quot;C&quot;
{
   int strncpy(...);
   int my_great_function();
};
</PRE>
<H3>Creating a C Function in C++ </H3><PRE>extern &quot;C&quot; void
a_c_function_in_cplusplus(int a)
{
}
</PRE>
<H3><I>__cplusplus</I> Preprocessor Directive </H3>If you have code that must 
  compile in a C and C++ environment then you must use the <I>__cplusplus</I> 
  preprocessor directive. For example: 
<P><PRE>#ifdef __cplusplus

extern &quot;C&quot; some_function();

#else

extern some_function();

#endif
</PRE>
<HR>
<A name=nomagic></A>
<H2>No Magic Numbers </H2>A magic number is a bare naked number used in source 
code. It's magic because no-one has a clue what it means including the author 
inside 3 months. For example: 
<P><PRE>if      (22 == foo) { start_thermo_nuclear_war(); }
else if (19 == foo) { refund_lotso_money(); }
else if (16 == foo) { infinite_loop(); }
else                { cry_cause_im_lost(); }
</PRE>In the above example what do 22 and 19 mean? If there was a number change 
or the numbers were just plain wrong how would you know? <P? <P thing. a such do 
never would they or code maintain to had has nor environment team in worked 
programmer Such else. anything than more amateur an as marks numbers magic of 
use Heavy>Instead of magic numbers use a real name that means something. You can 
use <I>#define</I> or constants or enums as names. Which one is a design choice. 
For example: <PRE>#define   PRESIDENT_WENT_CRAZY  (22)
const int WE_GOOFED= 19;
enum  {
   THEY_DIDNT_PAY= 16
};

if      (PRESIDENT_WENT_CRAZY == foo) { start_thermo_nuclear_war(); }
else if (WE_GOOFED            == foo) { refund_lotso_money(); }
else if (THEY_DIDNT_PAY       == foo) { infinite_loop(); }
else                                  { happy_days_i_know_why_im_here(); }
</PRE>Now isn't that better? The const and enum options are preferable because 
when debugging the debugger has enough information to display both the value and 
the label. The #define option just shows up as a number in the debugger which is 
very inconvenient. The const option has the downside of allocating memory. Only 
you know if this matters for your application. 
<P>
<HR>
<A name=errorret></A>
<H2>Error Return Check Policy </H2>
<UL>
  <LI>Check every system call for an error return, unless you know you wish to 
  ignore errors. For example, <I>printf</I> returns an error code but rarely 
  would you check for its return code. In which case you can cast the return to 
  <B>(void)</B> if you really care. 
  <LI>Include the system error text for every system error message. 
  <LI>Check every call to malloc or realloc unless you know your versions of 
  these calls do the right thing. You might want to have your own wrapper for 
  these calls, including new, so you can do the right thing always and 
  developers don't have to make memory checks everywhere. </LI></UL>
<A name=embedded>
<P>

  </h4>

  </BODY></HTML>
Markdown

<center>

# _C_ <span style="font-weight: 400">Coding Standard</span>

#### Adapted from [http://www.possibility.com/Cpp/CppCodingStandard.html](http://www.possibility.com/Cpp/CppCodingStandard.html) and NetBSD's style guidelines

</center>

For the C++ coding standards click [here](CppCodingStandard.html)

* * *

# Contents

1.  [**Names**](#names)
    *   _(important recommendations below)_
    *   [Include Units in Names](#units)
    *   [Structure Names](#classnames)
    *   [C File Extensions](#fext)
    *   _(other suggestions below)_
    *   [Make Names Fit](#descriptive)
    *   [Variable Names on the Stack](#stacknames)
    *   [Pointer Variables](#pnames)
    *   [Global Constants](#gconstants)
    *   [Enum Names](#enames)
    *   [#define and Macro Names](#mnames)<a name="docidx"></a>
2.  [**Formatting**](#formatting)
    *   _(important recommendations below)_
    *   [Brace _{}_ Policy](#brace)
    *   [Parens _()_ with Key Words and Functions Policy](#parens)
    *   [A Line Should Not Exceed 78 Characters](#linelen)
    *   [_If Then Else_ Formatting](#ifthen)
    *   [_switch_ Formatting](#switch)
    *   [Use of _goto,continue,break_ and _?:_](#goto)
    *   _(other suggestions below)_
    *   [One Statement Per Line](#one)
3.  [**Documentation**](#documentation)
    *   _(important recommendations below)_
    *   [Comments Should Tell a Story](#cstas)
    *   [Document Decisions](#cdd)
    *   [Use Headers](#cuh)
    *   [Make Gotchas Explicit](#mge)
    *   [Commenting function declarations](#cdef)
    *   _(other suggestions below)_
    *   [Include Statement Documentation](#idoc)
4.  [**Complexity Management**](#complexity)
    *   [Layering](#layering)
5.  [**Miscellaneous**](#misc)
    *   _(important recommendations below)_
    *   [Use Header File Guards](#guards)
    *   [Mixing C and C++](#callc)
    *   _(other suggestions below)_
    *   [Initialize all Variables](#initvar)
    *   [Be Const Correct](#const)
    *   [Short Functions](#shortmethods)
    *   [No Magic Numbers](#nomagic)
    *   [Error Return Check Policy](#errorret)
    *   [To Use Enums or Not to Use Enums](#useenums)
    *   [Macros](#macros)
    *   [Do Not Default If Test to Non-Zero](#nztest)
    *   [Usually Avoid Embedded Assignments](#eassign)
    *   [Commenting Out Large Code Blocks](#if0)
    *   [Use #if Not #ifdef](#ifdef)
    *   [Miscellaneous](#misc)
    *   [No Data Definitions in Header Files](#nodef)

* * *

<a name="names"></a>

# Names

<a name="descriptive"></a>

* * *

## Make Names Fit

Names are the heart of programming. In the past people believed knowing someone's true name gave them magical power over that person. If you can think up the true name for something, you give yourself and the people coming after power over the code. Don't laugh!

A name is the result of a long deep thought process about the ecology it lives in. Only a programmer who understands the system as a whole can create a name that "fits" with the system. If the name is appropriate everything fits together naturally, relationships are clear, meaning is derivable, and reasoning from common human expectations works as expected.

If you find all your names could be Thing and DoIt then you should probably revisit your design.

* * *

## Function Names

*   Usually every function performs an action, so the name should make clear what it does: check_for_errors() instead of error_check(), dump_data_to_file() instead of data_file(). This will also make functions and data objects more distinguishable.

    Structs are often nouns. By making function names verbs and following other naming conventions programs can be read more naturally.

*   Suffixes are sometimes useful:
    *   _max_ - to mean the maximum value something can have.
    *   _cnt_ - the current count of a running count variable.
    *   _key_ - key value.

    For example: retry_max to mean the maximum number of retries, retry_cnt to mean the current retry count.

*   Prefixes are sometimes useful:
    *   _is_ - to ask a question about something. Whenever someone sees _Is_ they will know it's a question.
    *   _get_ - get a value.
    *   _set_ - set a value.

    For example: is_hit_retry_limit.

<a name="units"></a>

* * *

## Include Units in Names

If a variable represents time, weight, or some other unit then include the unit in the name so developers can more easily spot problems. For example:

<pre>uint32 timeout_msecs;
uint32 my_weight_lbs;

</pre>

* * *

<a name="classnames"></a>

## Structure Names

*   Use underbars ('_') to separate name components
*   When declaring variables in structures, declare them organized by use in a manner to attempt to minimize memory wastage because of compiler alignment issues, then by size, and then by alphabetical order. E.g, don't use ``int a; char *b; int c; char *d''; use ``int a; int b; char *c; char *d''. Each variable gets its own type and line, although an exception can be made when declaring bitfields (to clarify that it's part of the one bitfield). Note that the use of bitfields in general is discouraged. Major structures should be declared at the top of the file in which they are used, or in separate header files, if they are used in multiple source files. Use of the structures should be by separate declarations and should be "extern" if they are declared in a header file. It may be useful to use a meaningful prefix for each member name. E.g, for ``struct softc'' the prefix could be ``sc_''.

### Example

<pre>   
struct foo {
	struct foo *next;	/* List of active foo */
	struct mumble amumble;	/* Comment for mumble */
	int bar;
	unsigned int baz:1,	/* Bitfield; line up entries if desired */
		     fuz:5,
		     zap:2;
	uint8_t flag;
};
struct foo *foohead;		/* Head of global foo list */

</pre>

* * *

<a name="stacknames"></a>

## Variable Names on the Stack

*   use all lower case letters
*   use '_' as the word separator.

### Justification

*   With this approach the scope of the variable is clear in the code.
*   Now all variables look different and are identifiable in the code.

### Example

<pre>   
   int handle_error (int error_number) {
      int            error= OsErr();
      Time           time_of_error;
      ErrorProcessor error_processor;
   }
</pre>

* * *

<a name="pnames"></a>

## Pointer Variables

*   place the _*_ close to the variable name not pointer type

### Example

<pre>  char *name= NULL;

  char *name, address; 
</pre>

* * *

## Global Variables

*   Global variables should be prepended with a 'g_'.
*   Global variables should be avoided whenever possible.

### Justification

*   It's important to know the scope of a variable.

### Example

<pre>    Logger  g_log;
    Logger* g_plog;
</pre>

* * *

<a name="gconstants"></a>

## Global Constants

*   Global constants should be all caps with '_' separators.

### Justification

It's tradition for global constants to named this way. You must be careful to not conflict with other global _#define_s and enum labels.

### Example

<pre>    const int A_GLOBAL_CONSTANT= 5;</pre>

<a name="snames"></a>

* * *

<a name="mnames"></a>

## #define and Macro Names

*   Put #defines and macros in all upper using '_' separators. Macros are capitalized, parenthesized, and should avoid side-effects. Spacing before and after the macro name may be any whitespace, though use of TABs should be consistent through a file. If they are an inline expansion of a function, the function is defined all in lowercase, the macro has the same name all in uppercase. If the macro is an expression, wrap the expression in parenthesis. If the macro is more than a single statement, use ``do { ... } while (0)'', so that a trailing semicolon works. Right-justify the backslashes; it makes it easier to read.

### Justification

This makes it very clear that the value is not alterable and in the case of macros, makes it clear that you are using a construct that requires care.

Some subtle errors can occur when macro names and enum labels use the same name.

### Example

<pre>#define MAX(a,b) blah
#define IS_ERR(err) blah
#define	MACRO(v, w, x, y)						\
do {									\
	v = (x) + (y);							\
	w = (y) + 2;							\
} while (0)
</pre>

<a name="cnames"></a>

* * *

<a name="enames"></a>

## Enum Names

### Labels All Upper Case with '_' Word Separators

This is the standard rule for enum labels. No comma on the last element.

#### Example

<pre>   enum PinStateType {
      PIN_OFF,
      PIN_ON
   };
</pre>

### Make a Label for an Error State

It's often useful to be able to say an enum is not in any of its _valid_ states. Make a label for an uninitialized or error state. Make it the first label if possible.

#### Example

<pre>enum { STATE_ERR,  STATE_OPEN, STATE_RUNNING, STATE_DYING};</pre>

<a name="req"></a>

* * *

<a name="formatting"></a>

# Formatting

* * *

<a name="brace"></a>

## Brace Placement

Of the three major brace placement strategies one is recommended:

<pre>   if (condition) {      while (condition) {
      ...                   ...
   }                     }
</pre>

* * *

## When Braces are Needed

All if, while and do statements must either have braces or be on a single line.

### Always Uses Braces Form

All if, while and do statements require braces even if there is only a single statement within the braces. For example:

<pre>if (1 == somevalue) {
   somevalue = 2;
}
</pre>

#### Justification

It ensures that when someone adds a line of code later there are already braces and they don't forget. It provides a more consistent look. This doesn't affect execution speed. It's easy to do.

### One Line Form

<pre>if (1 == somevalue) somevalue = 2;
</pre>

#### Justification

It provides safety when adding new lines while maintainng a compact readable form.

* * *

## Add Comments to Closing Braces

Adding a comment to closing braces can help when you are reading code because you don't have to find the begin brace to know what is going on.

<pre>while(1) {
   if (valid) {

   } /* if valid */
   else {
   } /* not valid */

} /* end forever */
</pre>

* * *

## Consider Screen Size Limits

Some people like blocks to fit within a common screen size so scrolling is not necessary when reading code.  

* * *

<a name="parens"></a>

## Parens _()_ with Key Words and Functions Policy

*   Do not put parens next to keywords. Put a space between.
*   Do put parens next to function names.
*   Do not use parens in return statements when it's not necessary.

### Justification

*   Keywords are not functions. By putting parens next to keywords keywords and function names are made to look alike.

### Example

<pre>    if (condition) {
    }

    while (condition) {
    }

    strcpy(s, s1);

    return 1;</pre>

* * *

<a name="linelen"></a>

## A Line Should Not Exceed 78 Characters

*   Lines should not exceed 78 characters.

## Justification

*   Even though with big monitors we stretch windows wide our printers can only print so wide. And we still need to print code.
*   The wider the window the fewer windows we can have on a screen. More windows is better than wider windows.
*   We even view and print diff output correctly on all terminals and printers.

* * *

<a name="ifthen"></a>

## _If Then Else_ Formatting

### Layout

It's up to the programmer. Different bracing styles will yield slightly different looks. One common approach is:

<pre>   if (condition) {
   } else if (condition) {
   } else {
   }
</pre>

If you have _else if_ statements then it is usually a good idea to always have an else block for finding unhandled cases. Maybe put a log message in the else even if there is no corrective action taken.

### Condition Format

Always put the constant on the left hand side of an equality/inequality comparison. For example:

if ( 6 == errorNum ) ...

One reason is that if you leave out one of the = signs, the compiler will find the error for you. A second reason is that it puts the value you are looking for right up front where you can find it instead of buried at the end of your expression. It takes a little time to get used to this format, but then it really gets useful.

* * *

<a name="switch"></a>

## _switch_ Formatting

*   Falling through a case statement into the next case statement shall be permitted as long as a comment is included.
*   The _default_ case should always be present and trigger an error if it should not be reached, yet is reached.
*   If you need to create variables put all the code in a block.

### Example

<pre>   switch (...)
   {
      case 1:
         ...
      /* comments */

      case 2:
      {        
         int v;
         ...
      }
      break;

      default:
   }
</pre>

* * *

<a name="goto"></a>

## Use of _goto,continue,break_ and _?:_

### Goto

Goto statements should be used sparingly, as in any well-structured code. The goto debates are boring so we won't go into them here. The main place where they can be usefully employed is to break out of several levels of switch, for, and while nesting, although the need to do such a thing may indicate that the inner constructs should be broken out into a separate function, with a success/failure return code.

<pre> <tt>for (...) {
      while (...) {
      ...
         if (disaster) {
            goto error;</tt></pre>

<pre>         } <tt>}
   }
   ...
error:
   clean up the mess</tt> </pre>

When a goto is necessary the accompanying label should be alone on a line and to the left of the code that follows. The goto should be commented (possibly in the block header) as to its utility and purpose.<a name="contbreak"></a>

### Continue and Break

Continue and break are really disguised gotos so they are covered here.

Continue and break like goto should be used sparingly as they are magic in code. With a simple spell the reader is beamed to god knows where for some usually undocumented reason.

The two main problems with continue are:

*   It may bypass the test condition
*   It may bypass the increment/decrement expression

Consider the following example where both problems occur:

<pre>while (TRUE) {
   ...
   /* A lot of code */
   ...
   if (/* some condition */) {
      continue;
   }
   ...
   /* A lot of code */
   ...
   if ( i++ > STOP_VALUE) break;
}
</pre>

Note: "A lot of code" is necessary in order that the problem cannot be caught easily by the programmer.

From the above example, a further rule may be given: Mixing continue with break in the same loop is a sure way to disaster.

### ?:

The trouble is people usually try and stuff too much code in between the _?_ and _:_. Here are a couple of clarity rules to follow:

*   Put the condition in parens so as to set it off from other code
*   If possible, the actions for the test should be simple functions.
*   Put the action for the then and else statement on a separate line unless it can be clearly put on one line.

### Example

<pre>   (condition) ? funct1() : func2();

   or

   (condition)
      ? long statement
      : another long statement;</pre>

<a name="aligndecls"></a>

* * *

<a name="one"></a>

## One Statement Per Line

There should be only one statement per line unless the statements are very closely related.

The reasons are:

1.  The code is easier to read. Use some white space too. Nothing better than to read code that is one line after another with no white space or comments.

### One Variable Per Line

Related to this is always define one variable per line:

<pre>**Not:**
char **a, *x;

**Do**:
char **a = 0;  /* add doc */
char  *x = 0;  /* add doc */
</pre>

The reasons are:

1.  Documentation can be added for the variable on the line.
2.  It's clear that the variables are initialized.
3.  Declarations are clear which reduces the probablity of declaring a pointer when you meant to declare just a char.

* * *

<a name="useenums"></a>

## To Use Enums or Not to Use Enums

C allows constant variables, which should deprecate the use of enums as constants. Unfortunately, in most compilers constants take space. Some compilers will remove constants, but not all. Constants taking space precludes them from being used in tight memory environments like embedded systems. Workstation users should use constants and ignore the rest of this discussion.

In general enums are preferred to _#define_ as enums are understood by the debugger.

Be aware enums are not of a guaranteed size. So if you have a type that can take a known range of values and it is transported in a message you can't use an enum as the type. Use the correct integer size and use constants or _#define_. Casting between integers and enums is very error prone as you could cast a value not in the enum.

* * *

<a name="guards"></a>

## Use Header File Guards

Include files should protect against multiple inclusion through the use of macros that "guard" the files. Note that for C++ compatibility and interoperatibility reasons, do **not** use underscores '_' as the first or last character of a header guard (see below)

<pre>#ifndef sys_socket_h
  #define sys_socket_h  /* NOT _sys_socket_h_ */
  #endif 
  </pre>

* * *

<a name="macros"></a>

# Macros

## Don't Turn C into Pascal

Don't change syntax via macro substitution. It makes the program unintelligible to all but the perpetrator.

## Replace Macros with Inline Functions

In C macros are not needed for code efficiency. Use inlines. However, macros for small functions are ok.

### Example

<pre>#define  MAX(x,y)	(((x) > (y) ? (x) : (y))	// Get the maximum
</pre>

The macro above can be replaced for integers with the following inline function with no loss of efficiency:

<pre>   inline int 
   max(int x, int y) {
      return (x > y ? x : y);
   }
</pre>

## Be Careful of Side Effects

Macros should be used with caution because of the potential for error when invoked with an expression that has side effects.

### Example

<pre>   MAX(f(x),z++);
</pre>

## Always Wrap the Expression in Parenthesis

When putting expressions in macros always wrap the expression in parenthesis to avoid potential communitive operation abiguity.

### Example

<pre>#define ADD(x,y) x + y

must be written as 

#define ADD(x,y) ((x) + (y))
</pre>

## Make Macro Names Unique

Like global variables macros can conflict with macros from other packages.

1.  Prepend macro names with package names.
2.  Avoid simple and common names like MAX and MIN.

* * *

<a name="initvar"></a>

# Initialize all Variables

*   You shall always initialize variables. Always. Every time. gcc with the flag -W may catch operations on uninitialized variables, but it may also not.

## Justification

*   More problems than you can believe are eventually traced back to a pointer or variable left uninitialized.

<a name="init"></a>

* * *

<a name="shortmethods"></a>

# Short Functions

*   Functions should limit themselves to a single page of code.

### Justification

*   The idea is that the each method represents a technique for achieving a single objective.
*   Most arguments of inefficiency turn out to be false in the long run.
*   True function calls are slower than not, but there needs to a thought out decision (see premature optimization).

* * *

<a name="docnull"></a>

# Document Null Statements

Always document a null body for a for or while statement so that it is clear that the null body is intentional and not missing code.

<pre> <tt>while (*dest++ = *src++)</tt> </pre>

<pre> <tt>{
      ;</tt> </pre>

<pre>   }</pre>

* * *

<a name="nztest"></a>

# Do Not Default If Test to Non-Zero

Do not default the test for non-zero, i.e.

<pre> <tt>if (FAIL != f())</tt> </pre>

is better than

<pre> <tt>if (f())</tt> </pre>

even though FAIL may have the value 0 which C considers to be false. An explicit test will help you out later when somebody decides that a failure return should be -1 instead of 0\. Explicit comparison should be used even if the comparison value will never change; e.g., **if (!(bufsize % sizeof(int)))** should be written instead as **if ((bufsize % sizeof(int)) == 0)** to reflect the numeric (not boolean) nature of the test. A frequent trouble spot is using strcmp to test for string equality, where the result should _never_ _ever_ be defaulted. The preferred approach is to define a macro _STREQ_.

<pre> <tt>#define STREQ(a, b) (strcmp((a), (b)) == 0)</tt> </pre>

Or better yet use an inline method:

<pre> <tt>inline bool
   string_equal(char* a, char* b)
   {
      (strcmp(a, b) == 0) ? return true : return false;
	  Or more compactly:
      return (strcmp(a, b) == 0);
   }</tt> </pre>

Note, this is just an example, you should really use the standard library string type for doing the comparison.

The non-zero test is often defaulted for predicates and other functions or expressions which meet the following restrictions:

*   Returns 0 for false, nothing else.
*   Is named so that the meaning of (say) a **true** return is absolutely obvious. Call a predicate is_valid(), not check_valid().

<a name="boolean"></a>

* * *

<a name="eassign"></a>

# Usually Avoid Embedded Assignments

There is a time and a place for embedded assignment statements. In some constructs there is no better way to accomplish the results without making the code bulkier and less readable.

<pre> <tt>while (EOF != (c = getchar())) {
      process the character
   }</tt> </pre>

The ++ and -- operators count as assignment statements. So, for many purposes, do functions with side effects. Using embedded assignment statements to improve run-time performance is also possible. However, one should consider the tradeoff between increased speed and decreased maintainability that results when embedded assignments are used in artificial places. For example,

<pre> <tt>a = b + c;
   d = a + r;</tt> </pre>

should not be replaced by

<pre> <tt>d = (a = b + c) + r;</tt> </pre>

even though the latter may save one cycle. In the long run the time difference between the two will decrease as the optimizer gains maturity, while the difference in ease of maintenance will increase as the human memory of what's going on in the latter piece of code begins to fade.

* * *

<a name="documentation"></a>

# Documentation

* * *

<a name="cstas"></a>

## Comments Should Tell a Story

Consider your comments a story describing the system. Expect your comments to be extracted by a robot and formed into a man page. Class comments are one part of the story, method signature comments are another part of the story, method arguments another part, and method implementation yet another part. All these parts should weave together and inform someone else at another point of time just exactly what you did and why.

* * *

<a name="cdd"></a>

## Document Decisions

Comments should document decisions. At every point where you had a choice of what to do place a comment describing which choice you made and why. Archeologists will find this the most useful information.<a name="cuh"></a>

* * *

## Use Headers

Use a document extraction system like [Doxygen](http://www.doxygen.org).

These headers are structured in such a way as they can be parsed and extracted. They are not useless like normal headers. So take time to fill them out. If you do it right once no more documentation may be necessary.

* * *

## Comment Layout

Each part of the project has a specific comment layout. [Doxygen](http://www.doxygen.org) has the recommended format for the comment layouts.

* * *

<a name="mge"></a>

## Make Gotchas Explicit

Explicitly comment variables changed out of the normal control flow or other code likely to break during maintenance. Embedded keywords are used to point out issues and potential problems. Consider a robot will parse your comments looking for keywords, stripping them out, and making a report so people can make a special effort where needed.

### Gotcha Keywords

*   **@author:**  
    specifies the author of the module
*   **@version:**  
    specifies the version of the module
*   **@param:**  
    specifies a parameter into a function
*   **@return:**  
    specifies what a function returns
*   **@deprecated:**  
    says that a function is not to be used anymore
*   **@see:**  
    creates a link in the documentation to the file/function/variable to consult to get a better understanding on what the current block of code does.
*   **@todo:**  
    what remains to be done
*   **@bug:**  
    report a bug found in the piece of code

### Gotcha Formatting

*   Make the gotcha keyword the first symbol in the comment.
*   Comments may consist of multiple lines, but the first line should be a self-containing, meaningful summary.
*   The writer's name and the date of the remark should be part of the comment. This information is in the source repository, but it can take a quite a while to find out when and by whom it was added. Often gotchas stick around longer than they should. Embedding date information allows other programmer to make this decision. Embedding who information lets us know who to ask.

<a name="cdef"></a>

## Commenting function declarations

Functions headers should be in the file where they are declared. This means that most likely the functions will have a header in the .h file. However, functions like main() with no explicit prototype declaration in the .h file, should have a header in the .c file.<a name="cflayout"></a>

* * *

<a name="idoc"></a>

## Include Statement Documentation

Include statements should be documented, telling the user why a particular file was included.  
/*  
* Kernel include files come first.  
*/  
/* Non-local includes in brackets. */  
/*  
* If it's a network program, put the network include files next.  
* Group the includes files by subdirectory.  
*/  
/*  
* Then there's a blank line, followed by the /usr include files.  
* The /usr include files should be sorted!  
*/

* * *

<a name="layering"></a>

# Layering

* * *

Layering is the primary technique for reducing complexity in a system. A system should be divided into layers. Layers should communicate between adjacent layers using well defined interfaces. When a layer uses a non-adjacent layer then a layering violation has occurred.

A layering violation simply means we have dependency between layers that is not controlled by a well defined interface. When one of the layers changes code could break. We don't want code to break so we want layers to work only with other adjacent layers.

Sometimes we need to jump layers for performance reasons. This is fine, but we should know we are doing it and document appropriately.

* * *

<a name="misc"></a>

# Miscellaneous

* * *

## General advice

This section contains some miscellaneous do's and don'ts.

*   Don't use floating-point variables where discrete values are needed. Using a float for a loop counter is a great way to shoot yourself in the foot. Always test floating-point numbers as <= or >=, never use an exact comparison (== or !=).
*   Compilers have bugs. Common trouble spots include structure assignment and bit fields. You cannot generally predict which bugs a compiler has. You could write a program that avoids all constructs that are known broken on all compilers. You won't be able to write anything useful, you might still encounter bugs, and the compiler might get fixed in the meanwhile. Thus, you should write ``around'' compiler bugs only when you are forced to use a particular buggy compiler.
*   Do not rely on automatic beautifiers. The main person who benefits from good program style is the programmer him/herself, and especially in the early design of handwritten algorithms or pseudo-code. Automatic beautifiers can only be applied to complete, syntactically correct programs and hence are not available when the need for attention to white space and indentation is greatest. Programmers can do a better job of making clear the complete visual layout of a function or file, with the normal attention to detail of a careful programmer (in other words, some of the visual layout is dictated by intent rather than syntax and beautifiers cannot read minds). Sloppy programmers should learn to be careful programmers instead of relying on a beautifier to make their code readable. Finally, since beautifiers are non-trivial programs that must parse the source, a sophisticated beautifier is not worth the benefits gained by such a program. Beautifiers are best for gross formatting of machine-generated code.
*   Accidental omission of the second ``='' of the logical compare is a problem. The following is confusing and prone to error.

    <pre>        if (abool= bbool) { ... }
         </pre>

    Does the programmer really mean assignment here? Often yes, but usually no. The solution is to just not do it, an inverse Nike philosophy. Instead use explicit tests and avoid assignment with an implicit test. The recommended form is to do the assignment before doing the test:

    <pre> <tt>abool= bbool;
           if (abool) { ... }</tt> </pre>

*   Modern compilers will put variables in registers automatically. Use the register sparingly to indicate the variables that you think are most critical. In extreme cases, mark the 2-4 most critical values as register and mark the rest as REGISTER. The latter can be #defined to register on those machines with many registers.

* * *

<a name="const"></a>

## Be Const Correct

C provides the _const_ key word to allow passing as parameters objects that cannot change to indicate when a method doesn't modify its object. Using const in all the right places is called "const correctness." It's hard at first, but using const really tightens up your coding style. Const correctness grows on you.<a name="streams"></a>

* * *

<a name="ifdef"></a>

## Use #if Not #ifdef

Use #if MACRO not #ifdef MACRO. Someone might write code like:

<pre>#ifdef DEBUG
        temporary_debugger_break();
#endif
</pre>

Someone else might compile the code with turned-of debug info like:

<pre>cc -c lurker.cc -DDEBUG=0
</pre>

Alway use #if, if you have to use the preprocessor. This works fine, and does the right thing, even if DEBUG is not defined at all (!)

<pre>#if DEBUG
        temporary_debugger_break();
#endif
</pre>

If you really need to test whether a symbol is defined or not, test it with the defined() construct, which allows you to add more things later to the conditional without editing text that's already in the program:

<pre>#if !defined(USER_NAME)
 #define USER_NAME "john smith"
#endif
</pre>

* * *

<a name="if0"></a>

## Commenting Out Large Code Blocks

Sometimes large blocks of code need to be commented out for testing.

### Using #if 0

The easiest way to do this is with an #if 0 block:

<pre>   void 
   example()
   {
      great looking code

      #if 0
      lots of code
      #endif

      more code
    }
</pre>

You can't use **/**/** style comments because comments can't contain comments and surely a large block of your code will contain a comment, won't it?

Don't use #ifdef as someone can unknowingly trigger ifdefs from the compiler command line. <h3use descriptive="" macro="" names="" instead="" of="" 0="" <="" h3the="" problem="" with="" <b="">#if 0is that even day later you or anyone else has know idea why this code is commented out. Is it because a feature has been dropped? Is it because it was buggy? It didn't compile? Can it be added back? It's a mystery.</h3use>

### Use Descriptive Macro Names Instead of #if 0

<pre>#if NOT_YET_IMPLEMENTED  

#if OBSOLETE

#if TEMP_DISABLED 
</pre>

### Add a Comment to Document Why

Add a short comment explaining why it is not implemented, obsolete or temporarily disabled.<a name="accessor"></a><a name="personas"></a>

* * *

<a name="fext">

## File Extensions

In short: Use the _.h_ extension for header files and _.c_ for source files.

* * *

</a><a name="nodef"></a>

## No Data Definitions in Header Files

Do not put data definitions in header files. for example:

<pre>/* 
 * aheader.h 
 */
int x = 0;
</pre>

1.  It's bad magic to have space consuming code silently inserted through the innocent use of header files.
2.  It's not common practice to define variables in the header file so it will not occur to developers to look for this when there are problems.
3.  Consider defining the variable once in a .c file and use an extern statement to reference it.

* * *

<a name="callc"></a>

## Mixing C and C++

In order to be backward compatible with dumb linkers C++'s link time type safety is implemented by encoding type information in link symbols, a process called _name mangling_. This creates a problem when linking to C code as C function names are not mangled. When calling a C function from C++ the function name will be mangled unless you turn it off. Name mangling is turned off with the _extern "C"_ syntax. If you want to create a C function in C++ you must wrap it with the above syntax. If you want to call a C function in a C library from C++ you must wrap in the above syntax. Here are some examples:

### Calling C Functions from C++

<pre>extern "C" int strncpy(...);
extern "C" int my_great_function();
extern "C"
{
   int strncpy(...);
   int my_great_function();
};
</pre>

### Creating a C Function in C++

<pre>extern "C" void
a_c_function_in_cplusplus(int a)
{
}
</pre>

### ___cplusplus_ Preprocessor Directive

If you have code that must compile in a C and C++ environment then you must use the ___cplusplus_ preprocessor directive. For example:

<pre>#ifdef __cplusplus

extern "C" some_function();

#else

extern some_function();

#endif
</pre>

* * *

<a name="nomagic"></a>

## No Magic Numbers

A magic number is a bare naked number used in source code. It's magic because no-one has a clue what it means including the author inside 3 months. For example:

<pre>if      (22 == foo) { start_thermo_nuclear_war(); }
else if (19 == foo) { refund_lotso_money(); }
else if (16 == foo) { infinite_loop(); }
else                { cry_cause_im_lost(); }
</pre>

In the above example what do 22 and 19 mean? If there was a number change or the numbers were just plain wrong how would you know? <p? <p="" thing.="" a="" such="" do="" never="" would="" they="" or="" code="" maintain="" to="" had="" has="" nor="" environment="" team="" in="" worked="" programmer="" else.="" anything="" than="" more="" amateur="" an="" as="" marks="" numbers="" magic="" of="" use="" heavy="">Instead of magic numbers use a real name that means something. You can use _#define_ or constants or enums as names. Which one is a design choice. For example:

<pre>#define   PRESIDENT_WENT_CRAZY  (22)
const int WE_GOOFED= 19;
enum  {
   THEY_DIDNT_PAY= 16
};

if      (PRESIDENT_WENT_CRAZY == foo) { start_thermo_nuclear_war(); }
else if (WE_GOOFED            == foo) { refund_lotso_money(); }
else if (THEY_DIDNT_PAY       == foo) { infinite_loop(); }
else                                  { happy_days_i_know_why_im_here(); }
</pre>

Now isn't that better? The const and enum options are preferable because when debugging the debugger has enough information to display both the value and the label. The #define option just shows up as a number in the debugger which is very inconvenient. The const option has the downside of allocating memory. Only you know if this matters for your application.

* * *

<a name="errorret"></a>

## Error Return Check Policy

*   Check every system call for an error return, unless you know you wish to ignore errors. For example, _printf_ returns an error code but rarely would you check for its return code. In which case you can cast the return to **(void)** if you really care.
*   Include the system error text for every system error message.
*   Check every call to malloc or realloc unless you know your versions of these calls do the right thing. You might want to have your own wrapper for these calls, including new, so you can do the right thing always and developers don't have to make memory checks everywhere.

<a name="embedded"></a></p?>
to-markdown is copyright © 2011-15 Dom Christie and is released under the MIT licence.
