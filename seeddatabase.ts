import { ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings } from "@langchain/google-genai";
import { StructuredOutputParser } from "@langchain/core/output_parsers";
import { MongoClient } from "mongodb";
import { MongoDBAtlasVectorSearch } from "@langchain/mongodb";
import { z } from "zod";
import "dotenv/config";

const client = new MongoClient(process.env.MONGO_URI as string);

const llm = new ChatGoogleGenerativeAI({
  model: "gemini-1.5-flash",
  temperature: 0.7,
  apiKey: process.env.GOOGLE_API_KEY,
});

const EmployeeSchema = z.object({
  employee_id: z.string(),
  first_name: z.string(),
  last_name: z.string(),
  date_of_birth: z.string(),
  address: z.object({
    street: z.string(),
    city: z.string(),
    state: z.string().nullable(),
    postal_code: z.string(),
    country: z.string(),
  }),
  contact_details: z.object({
    email: z.string().email(),
    phone_number: z.string(),
  }),
  job_details: z.object({
    job_title: z.string(),
    department: z.string(),
    hire_date: z.string(),
    employment_type: z.string(),
    salary: z.number(),
    currency: z.string(),
  }),
  work_location: z.object({
    nearest_office: z.string(),
    is_remote: z.boolean(),
  }),
  reporting_manager: z.string().nullable(),
  skills: z.array(z.string()),
  performance_reviews: z.array(
    z.object({
      review_date: z.string(),
      rating: z.number(),
      comments: z.string(),
    })
  ),
  benefits: z.object({
    health_insurance: z.string(),
    retirement_plan: z.string(),
    paid_time_off: z.number(),
  }),
  emergency_contact: z.object({
    name: z.string(),
    relationship: z.string(),
    phone_number: z.string(),
  }),
  notes: z.string(),
});

type Employee = z.infer<typeof EmployeeSchema>;

const parser = StructuredOutputParser.fromZodSchema(z.array(EmployeeSchema));

async function generateSyntheticData(): Promise<Employee[]> {
  const prompt = `Generate exactly 10 fictional employee records as a valid JSON array. Return ONLY the JSON array, no markdown formatting, no explanations.

Each employee record must have these exact fields with correct data types:
- employee_id: "E001" (string)
- first_name: "John" (string)  
- last_name: "Doe" (string)
- date_of_birth: "1990-01-15" (string in YYYY-MM-DD format)
- address: object with:
  - street: "123 Main St" (string)
  - city: "New York" (string) 
  - state: "NY" (string) or null for non-US
  - postal_code: "10001" (string)
  - country: "USA" (string)
- contact_details: object with:
  - email: "john@example.com" (string)
  - phone_number: "555-123-4567" (string)
- job_details: object with:
  - job_title: "Software Engineer" (string)
  - department: "Engineering" (string)
  - hire_date: "2022-01-15" (string in YYYY-MM-DD format)
  - employment_type: "Full-time" (string)
  - salary: 75000 (number, no quotes)
  - currency: "USD" (string)
- work_location: object with:
  - nearest_office: "New York" (string)
  - is_remote: false (boolean, no quotes)
- reporting_manager: "Jane Smith" (string) or null
- skills: ["Python", "React"] (array of strings)
- performance_reviews: array of objects with:
  - review_date: "2023-01-15" (string)
  - rating: 4.5 (number, no quotes)
  - comments: "Great work" (string)
- benefits: object with:
  - health_insurance: "Blue Cross" (string)
  - retirement_plan: "401k" (string)  
  - paid_time_off: 20 (number, no quotes)
- emergency_contact: object with:
  - name: "Jane Doe" (string)
  - relationship: "Spouse" (string)
  - phone_number: "555-987-6543" (string)
- notes: "Excellent employee" (string)

Return only valid JSON starting with [ and ending with ].`;

  console.log("Generating synthetic data...");

  const response = await llm.invoke(prompt);
  let content = response.content as string;
  
  // More aggressive cleanup
  content = content.trim();
  
  // Remove common markdown patterns
  content = content.replace(/```json\s*/g, '');
  content = content.replace(/\s*```$/g, '');
  content = content.replace(/^```\s*/g, '');
  content = content.replace(/\s*```$/g, '');
  
  // Remove any leading/trailing non-JSON content
  const jsonStart = content.indexOf('[');
  const jsonEnd = content.lastIndexOf(']');
  
  if (jsonStart !== -1 && jsonEnd !== -1) {
    content = content.substring(jsonStart, jsonEnd + 1);
  }
  
  console.log("Cleaned content preview:", content.substring(0, 500) + "...");
  
  try {
    // First try to parse as JSON to validate structure
    const jsonData = JSON.parse(content);
    console.log("JSON parsed successfully, validating with schema...");
    
    // Then use the structured parser with better error reporting
    return parser.parse(content);
  } catch (parseError) {
    console.error("Parse error:", parseError);
    
    // Try to parse JSON first to see the raw data
    try {
      const rawData = JSON.parse(content);
      console.log("Raw JSON data (first record):", JSON.stringify(rawData[0], null, 2));
    } catch (jsonError) {
      console.error("JSON syntax error:", jsonError);
    }
    
    console.error("Content that failed to parse:", content.substring(0, 1000));
    throw new Error(`Failed to parse generated data: ${parseError.message}`);
  }
}

async function createEmployeeSummary(employee: Employee): Promise<string> {
  return new Promise((resolve) => {
    const jobDetails = `${employee.job_details.job_title} in ${employee.job_details.department}`;
    const skills = employee.skills.join(", ");
    const performanceReviews = employee.performance_reviews
      .map(
        (review) =>
          `Rated ${review.rating} on ${review.review_date}: ${review.comments}`
      )
      .join(" ");
    const basicInfo = `${employee.first_name} ${employee.last_name}, born on ${employee.date_of_birth}`;
    const workLocation = `Works at ${employee.work_location.nearest_office}, Remote: ${employee.work_location.is_remote}`;
    const notes = employee.notes;

    const summary = `${basicInfo}. Job: ${jobDetails}. Skills: ${skills}. Reviews: ${performanceReviews}. Location: ${workLocation}. Notes: ${notes}`;

    resolve(summary);
  });
}

async function seedDatabase(): Promise<void> {
  try {
    await client.connect();
    await client.db("admin").command({ ping: 1 });
    console.log("Pinged your deployment. You successfully connected to MongoDB!");

    const db = client.db("hr_database");
    const collection = db.collection("employees");

    await collection.deleteMany({});
    
    const syntheticData = await generateSyntheticData();

    const recordsWithSummaries = await Promise.all(
      syntheticData.map(async (record) => ({
        pageContent: await createEmployeeSummary(record),
        metadata: {...record},
      }))
    );
    
    for (const record of recordsWithSummaries) {
      await MongoDBAtlasVectorSearch.fromDocuments(
        [record],
        new GoogleGenerativeAIEmbeddings({
          apiKey: process.env.GOOGLE_API_KEY,
          model: "text-embedding-004",
        }),
        {
          collection,
          indexName: "vector_index",
          textKey: "embedding_text",
          embeddingKey: "embedding",
        }
      );

      console.log("Successfully processed & saved record:", record.metadata.employee_id);
    }

    console.log("Database seeding completed");

  } catch (error) {
    console.error("Error seeding database:", error);
  } finally {
    await client.close();
  }
}

seedDatabase().catch(console.error);