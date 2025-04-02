// / Import the os module
const os = require('os');
// Get information about the operating system
console.log('Platform:', os.platform());
console.log('Architecture:', os.arch());
console.log(`Total Memory: ${os.totalmem()}`);
console.log(`Free Memory: ${os.freemem()}`);
console.log(`CPU Cores: ${os.cpus()}`);
